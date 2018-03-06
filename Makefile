PWD = $(shell pwd)
default:
	make build
	make run
build:
	docker build \
		-t dmitry_test_model .
run:
	docker run \
		-v $(PWD):/apply_model \
		--rm \
		-it \
		-e LC_ALL=C.UTF-8 \
		-e LANG=C.UTF-8 \
		-w /apply_model \
		 dmitry_test_model /bin/bash
