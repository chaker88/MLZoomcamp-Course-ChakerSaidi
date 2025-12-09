#!/bin/bash

# Correct variable assignments (no spaces around =)
ECR_URL=070462957699.dkr.ecr.us-east-1.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda
LOCAL_IMAGE=churn-prediction-lambda
REMOTE_IMAGE_TAG=${REPO_URL}:v1

# Login to ECR
aws ecr get-login-password \
    --region us-east-1 \
    | docker login \
    --username AWS \
    --password-stdin ${ECR_URL}

# Build, tag, and push Docker image
docker build -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Done!"
