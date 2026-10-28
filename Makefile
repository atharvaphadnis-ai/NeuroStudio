# NeuroStudio — build & verify
GO ?= go
BINARY := neurostudio
VERSION ?= $(shell git describe --tags --always 2>/dev/null || echo dev)

.PHONY: build run test vet fmt check smoke clean release

build:
	$(GO) build -ldflags "-s -w" -o bin/$(BINARY) .

run: build
	./bin/$(BINARY) --verbose

test:
	$(GO) test ./... -count=1

vet:
	$(GO) vet ./...

fmt:
	gofmt -w main.go internal/

check: vet
	@out=$$($(GO) fmt ./... 2>/dev/null); [ -z "$$out" ] || { echo "unformatted files:"; echo "$$out"; exit 1; }

smoke: build
	./scripts/smoke.sh

clean:
	rm -rf bin/

# release builds for the three desktop OSes (cpus: amd64 + arm64 for darwin)
release:
	GOOS=linux   GOARCH=amd64 $(GO) build -ldflags "-s -w" -o bin/$(BINARY)-linux-amd64 .
	GOOS=windows GOARCH=amd64 $(GO) build -ldflags "-s -w" -o bin/$(BINARY)-windows-amd64.exe .
	GOOS=darwin  GOARCH=amd64 $(GO) build -ldflags "-s -w" -o bin/$(BINARY)-darwin-amd64 .
	GOOS=darwin  GOARCH=arm64 $(GO) build -ldflags "-s -w" -o bin/$(BINARY)-darwin-arm64 .
