terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# required APIs
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

# gcs bucket
resource "google_storage_bucket" "artifacts" {
  project = var.project_id
  name = "${var.project_id}-iot-artifacts"
  location = var.region
  force_destroy = true # Set to false in a real production environment
}

# artifact registry
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

# grant necessary permissions to sa
resource "google_project_iam_member" "sa_permissions" {
  for_each = toset([
    "roles/storage.admin",
    "roles/artifactregistry.admin",
    "roles/run.admin",
    "roles/iam.serviceAccountUser",
    "roles/cloudsql.client"
  ])
  project = var.project_id
  role = each.key
  member = google_service_account.github_actions_sa.member
}

# VPC network for Cloud Run to Cloud SQL
resource "google_compute_network" "vpc_network" {
  project = var.project_id
  name = "mlops-vpc-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "vpc_subnetwork" {
  project = var.project_id
  name = "mlops-subnetwork"
  ip_cidr_range = "10.10.10.0/24"
  region = var.region
  network = google_compute_network.vpc_network.id
  private_ip_google_access = true
}

resource "google_vpc_access_connector" "vpc_connector" {
  project = var.project_id
  name = "mlops-vpc"
  region = var.region
  ip_cidr_range = "10.8.0.0/28"
  network = google_compute_network.vpc_network.id
  depends_on    = [google_project_service.apis]
}

# Cloud SQL & DB
resource "google_sql_database_instance" "mlflow_pg" {
  project = var.project_id
  name = "mlflow-pg"
  database_version = "POSTGRES_17"
  region = var.region

  settings {
    tier = "db-g1-small"
    availability_type = "REGIONAL"
    ip_configuration {
      ipv4_enabled = false
      private_network = google_compute_network.vpc_network.id
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "mlflow_db" {
  project = var.project_id
  instance = google_sql_database_instance.mlflow_pg.name
  name = "mlflow_db"
}

resource "google_sql_database" "prefect_db" {
  project = var.project_id
  instance = google_sql_database_instance.mlflow_pg.name
  name = "prefect_db"
}

resource "google_sql_user" "mlflow_user" {
  project = var.project_id
  instance = google_sql_database_instance.mlflow_pg.name
  name = "mlflow_user"
  password = var.db_password
}

resource "google_sql_user" "prefect_user" {
  project  = var.project_id
  instance = google_sql_database_instance.mlflow_pg.name
  name     = "prefect_user"
  password = var.prefect_db_password
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
      size = 50
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

  depends_on = [google_project_service.apis]
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
