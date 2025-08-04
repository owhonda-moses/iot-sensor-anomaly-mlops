variable "project_id" {
  description = "GCP project ID"
}

variable "region" {
  description = "GCP region"
}

variable "db_password" {
  description = "Password for 'mlflow_user' in Cloud SQL."
  type = string
  sensitive = true
}
