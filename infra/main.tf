terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# enable eequired APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "sqladmin.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "vpcaccess.googleapis.com",
    "servicenetworking.googleapis.com"
  ])
  project = var.project_id
  service = each.key
  disable_on_destroy = false
}

# gcs artifacts bucket
resource "google_storage_bucket" "artifacts" {
  project = var.project_id
  name = "${var.project_id}-iot-artifacts"
  location = var.region
  force_destroy = true
}

# artifact registry for docker image
resource "google_artifact_registry_repository" "docker_repo" {
  project = var.project_id
  location = var.region
  repository_id = "mlops-docker"
  format = "DOCKER"
}

# service account for CI/CD
resource "google_service_account" "github_actions_sa" {
  project = var.project_id
  account_id = "github-actions-sa"
  display_name = "GitHub Actions Service Account"
}

# permissions for CI/CD pipeline
resource "google_project_iam_member" "sa_cicd_permissions" {
  for_each = toset([
    "roles/storage.admin",            # manage GCS bucket
    "roles/artifactregistry.admin",   # push docker images
    "roles/run.admin",                # deploy to cloud run
    "roles/iam.serviceAccountUser",   # act as other SAs during deployment
    "roles/cloudsql.client",          # connect cloud run to cloud SQL
    "roles/cloudbuild.builds.editor"  # use gcloud builds submit
  ])
  project = var.project_id
  role = each.key
  member = google_service_account.github_actions_sa.member
}

# permissions for running cloud run services
data "google_project" "project" {}

resource "google_project_iam_member" "run_services_permissions" {
  for_each = toset([
    "roles/storage.objectAdmin", # allows services to write to GCS
    "roles/cloudsql.client"      # allows MLflow server to connect to Cloud SQL
  ])
  project = var.project_id
  role = each.key
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# vpc network for cloud run to cloud SQL
resource "google_compute_network" "vpc_network" {
  project = var.project_id
  name = "mlops-vpc-network"
  auto_create_subnetworks = false
}

resource "google_vpc_access_connector" "vpc_connector" {
  project = var.project_id
  name = "mlops-vpc"
  region = var.region
  ip_cidr_range = "10.8.0.0/28"
  network = google_compute_network.vpc_network.id
  depends_on = [google_project_service.apis["vpcaccess.googleapis.com"]]
}

# cloud SQL instance for mlflow
resource "google_sql_database_instance" "mlflow_pg" {
  project = var.project_id
  name = "mlflow-pg"
  database_version = "POSTGRES_17"
  region = var.region
  settings {
    tier = "db-g1-small"
  }
  depends_on = [google_project_service.apis["sqladmin.googleapis.com"]]
}

resource "google_sql_database" "mlflow_db" {
  project = var.project_id
  instance = google_sql_database_instance.mlflow_pg.name
  name = "mlflow_db"
}

resource "google_sql_user" "mlflow_user" {
  project = var.project_id
  instance = google_sql_database_instance.mlflow_pg.name
  name = "mlflow_user"
  password = var.db_password
}

# GCE VM for prefect server
resource "google_compute_instance" "prefect_server_vm" {
  project = var.project_id
  name = "prefect-server-vm"
  machine_type = "e2-micro"
  zone = "${var.region}-c"
  tags = ["prefect-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y curl
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    docker run -d --restart always -p 4200:4200 --name prefect-server \
      prefecthq/prefect:2-latest \
      prefect server start --host 0.0.0.0 --port 4200
  EOF
  depends_on = [google_project_service.apis["compute.googleapis.com"]]
}

# firewall rule for prefect UI
resource "google_compute_firewall" "allow_prefect_ui" {
  project = var.project_id
  name = "allow-prefect-ui"
  network = "default"
  direction = "INGRESS"
  source_ranges = ["0.0.0.0/0"]
  target_tags = ["prefect-server"]

  allow {
    protocol = "tcp"
    ports = ["4200"]
  }
}
