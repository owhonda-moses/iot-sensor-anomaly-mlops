terraform {
  backend "gcs" {
    bucket = "iot-artifacts"
    prefix = "iot-mlops/infra"
  }
}