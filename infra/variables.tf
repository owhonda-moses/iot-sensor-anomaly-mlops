variable "project_id" {
  description = "GCP project ID"
}

variable "region" {
  description = "GCP region"
}

variable "db_password" {
  description = "Password for the 'mlflow_user' in Cloud SQL."
  type = string
  sensitive = true
}

variable "prefect_db_password" {
  description = "Password for the 'prefect_user' in Cloud SQL."
  type = string
  sensitive = true
}
