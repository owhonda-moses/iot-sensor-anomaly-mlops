output "gce_vm_external_ip" {
  description = "External IP address of the Prefect Server VM."
  value = google_compute_instance.prefect_server_vm.network_interface[0].access_config[0].nat_ip
}

output "gcs_bucket_name" {
  description = "Name of the GCS artifacts bucket."
  value = google_storage_bucket.artifacts.name
}

output "artifact_registry_name" {
  description = "Full name of the Docker repo."
  value = google_artifact_registry_repository.docker_repo.name
}

output "cloud_sql_instance_connection_name" {
  description = "Connection name for the Cloud SQL instance for services to use."
  value = google_sql_database_instance.mlflow_pg.connection_name
}

output "github_service_account_email" {
  description = "Email of the created GitHub Actions service account."
  value = google_service_account.github_actions_sa.email
}
