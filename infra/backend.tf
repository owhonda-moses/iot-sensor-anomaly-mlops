terraform {
  backend "gcs" {
    bucket = "iot-artifacts"
    prefix = "iot-mlops/infra"
    credentials = "gcp-sa-key.json"
  }
}