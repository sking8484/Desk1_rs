include .env
# Export the service list
export SERVICE_LIST=./service-list.txt

build-images:
	for service in `cat $${SERVICE_LIST}`; do \
		rm -rf ./src/$${service}/Dockerfile ; \
		cp Dockerfile ./src/$${service} ; \
		docker build -f "./src/$${service}/Dockerfile" -t "$${service}-container" . --build-arg function=$${service} ; \
	done
	#docker compose build

create-ecr-repo: build-images
	for service in `cat $${SERVICE_LIST}`; do \
		aws ecr create-repository --repository-name $${service}-repo || true ; \
	done

publish: create-ecr-repo
	for service in `cat $${SERVICE_LIST}`; do \
		docker tag $${service}-container:latest $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:latest ; \
		aws ecr get-login-password | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com ; \
		docker push $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:latest ; \
	done

deploy: publish
	for service in `cat $${SERVICE_LIST}`; do \
		aws cloudformation deploy --stack-name $${service}-$(CFN_STACK_NAME) \
		--template-file ./templateFile.yml \
		--parameter-overrides imageUri=$(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:latest service=$${service}; \
	done

destroy:
	for service in `cat $${SERVICE_LIST}`; do \
		aws cloudformation delete-stack --stack-name $${service}-$(CFN_STACK_NAME); \
	done
