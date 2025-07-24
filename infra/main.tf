locals {
  required_apis = [
    "cloudresourcemanager.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
  ]
}

resource "google_project_service" "enable_apis" {
  for_each = toset(local.required_apis)
  project = var.project_id
  service = each.value
  disable_on_destroy  = false
}



provider "google" {
  project = var.project_id
  region = var.region
  credentials = file(var.credentials_file)
}

# Artifacts bucket for MLflow & Prefect
resource "google_storage_bucket" "artifacts" {
  name = "${var.project_id}-iot-artifacts"
  location = var.region
  force_destroy = true
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "docker_repo" {
  provider = google
  location = var.region
  repository_id = "mlops-docker"
  format = "DOCKER"
}

data "google_service_account" "github" {
  project = var.project_id
  account_id = "github-actions-sa"
}

locals {
  sa_email = data.google_service_account.github.email
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role = "roles/storage.admin"
  member = "serviceAccount:${local.sa_email}"
}

resource "google_project_iam_member" "artifact_admin" {
  project = var.project_id
  role = "roles/artifactregistry.admin"
  member  = "serviceAccount:${local.sa_email}"
}

resource "google_project_iam_member" "run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${local.sa_email}"
}

# (Optional) If you’ll deploy to GKE:
# resource "google_project_iam_member" "container_admin" { … }

output "sa_key_file" {
  value = var.credentials_file
}
output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}
output "bucket_name" {
  value = google_storage_bucket.artifacts.name
}

