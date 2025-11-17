PROJECT_ROOT ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(PROJECT_ROOT)/Dockerfile
IMAGE_NAME ?= esc50-audio-dev
CONTAINER_NAME ?= esc50-audio-dev
MOUNT_SRC ?= $(PROJECT_ROOT)
MOUNT_DST ?= /workspace
JUPYTER_PORT ?= 8888
TEST_CMD ?= pytest -q
DOCKER ?= docker

.PHONY: build rebuild bash notebook test clean rm-image

build:
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(PROJECT_ROOT)

rebuild:
	$(DOCKER) build --no-cache -t $(IMAGE_NAME) -f $(DOCKERFILE) $(PROJECT_ROOT)

bash:
	$(DOCKER) run --rm -it \
		--name $(CONTAINER_NAME) \
		-v $(MOUNT_SRC):$(MOUNT_DST) \
		-p $(JUPYTER_PORT):8888 \
		$(IMAGE_NAME) bash

notebook:
	$(DOCKER) run --rm -it \
		--name $(CONTAINER_NAME) \
		-v $(MOUNT_SRC):$(MOUNT_DST) \
		-p $(JUPYTER_PORT):8888 \
		$(IMAGE_NAME) bash -lc "cd $(MOUNT_DST) && jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='' --ServerApp.root_dir=$(MOUNT_DST) --allow-root"

test:
	$(DOCKER) run --rm \
		-v $(MOUNT_SRC):$(MOUNT_DST) \
		$(IMAGE_NAME) bash -lc "cd $(MOUNT_DST) && $(TEST_CMD)"

clean:
	-$(DOCKER) rm -f $(CONTAINER_NAME) 2>/dev/null || true

rm-image: clean
	-$(DOCKER) rmi $(IMAGE_NAME) 2>/dev/null || true
