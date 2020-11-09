#!/usr/bin/env bash
# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build kfp component. Assumes component name is the same as the directory it
# is in, for example, export_tf_metrics is the directory within components
# as well as the name in the tag.

# Run from the bin/ directory

COMPONENT=$1
TAG=$2

if [ -z "$TAG" ]; then
  TAG=latest
fi

PROJECT_ID=$(gcloud config get-value project)
SOURCE=../components/${COMPONENT} 

gcloud builds submit $SOURCE \
  --async \
  --tag=gcr.io/${PROJECT_ID}/${COMPONENT}:${TAG} \
  --machine-type=n1-highcpu-8