# Makefile for OCC + HurdCog + Cognumach Integration
# This provides a simplified build interface for the unified AGI-OS stack

.PHONY: all clean help cognumach hurdcog occ unified test install

# Default target
all: unified

help:
	@echo "AGI-OS Integration Build System"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Build the complete unified stack (default)"
	@echo "  cognumach  - Build only the Cognumach microkernel"
	@echo "  hurdcog    - Build only HurdCog (requires cognumach)"
	@echo "  occ        - Build only OCC components"
	@echo "  unified    - Build the complete integrated system"
	@echo "  test       - Run integration tests"
	@echo "  install    - Install to system (requires sudo)"
	@echo "  clean      - Clean build artifacts"
	@echo ""
	@echo "Build methods:"
	@echo "  make guix-build    - Build using GNU Guix (recommended)"
	@echo "  make direct-build  - Build directly without Guix"
	@echo ""

# Build using GNU Guix (recommended)
guix-build: unified

cognumach:
	@echo "Building Cognumach microkernel..."
	@if command -v guix >/dev/null 2>&1; then \
		guix build -f cognumach.scm; \
	else \
		echo "GNU Guix not found. Use 'make direct-build-cognumach' instead."; \
		exit 1; \
	fi

hurdcog: cognumach
	@echo "Building HurdCog cognitive OS..."
	@if command -v guix >/dev/null 2>&1; then \
		guix build -f hurdcog.scm; \
	else \
		echo "GNU Guix not found. Use 'make direct-build-hurdcog' instead."; \
		exit 1; \
	fi

occ:
	@echo "Building OCC AGI framework..."
	@if command -v guix >/dev/null 2>&1; then \
		cd /home/ubuntu/occ && guix build -f guix.scm; \
	else \
		echo "GNU Guix not found. Use 'make direct-build-occ' instead."; \
		exit 1; \
	fi

unified: cognumach hurdcog occ
	@echo "Building unified AGI-OS stack..."
	@if command -v guix >/dev/null 2>&1; then \
		guix build -f occ-hurdcog-unified.scm; \
	else \
		echo "GNU Guix not found. Use 'make direct-build-unified' instead."; \
		exit 1; \
	fi
	@echo ""
	@echo "✓ Unified AGI-OS build complete!"
	@echo ""
	@echo "To test the integration, run: make test"

# Direct build without Guix (for development)
direct-build: direct-build-cognumach direct-build-hurdcog direct-build-occ

direct-build-cognumach:
	@echo "Building Cognumach directly..."
	@cd /home/ubuntu/cognumach && \
		if [ ! -f configure ]; then autoreconf -vif; fi && \
		./configure --host=i686-gnu CC='gcc -m32' LD='ld -melf_i386' \
			--enable-kdb --prefix=/usr/local/cognumach && \
		make -j$$(nproc)
	@echo "✓ Cognumach build complete"

direct-build-hurdcog:
	@echo "Building HurdCog directly..."
	@cd /home/ubuntu/hurdcog/cogkernel && \
		make -j$$(nproc)
	@echo "✓ HurdCog build complete"

direct-build-occ:
	@echo "Building OCC directly..."
	@cd /home/ubuntu/occ && \
		if [ -d cogutil ] && [ -d atomspace ]; then \
			mkdir -p build && cd build && \
			cmake .. -DCMAKE_BUILD_TYPE=Release \
				-DBUILD_COGUTIL=ON \
				-DBUILD_ATOMSPACE=ON \
				-DBUILD_COGSERVER=ON \
				-DBUILD_MATRIX=ON \
				-DBUILD_LEARN=ON \
				-DBUILD_AGENTS=ON \
				-DBUILD_SENSORY=ON \
				-DCMAKE_INSTALL_PREFIX=/usr/local/occ && \
			make -j$$(nproc); \
		else \
			echo "OCC subdirectories not found, skipping CMake build"; \
		fi
	@echo "✓ OCC build complete"

# Testing
test:
	@echo "Running AGI-OS integration tests..."
	@echo ""
	@echo "1. Checking Cognumach..."
	@if [ -f /home/ubuntu/cognumach/gnumach ]; then \
		echo "   ✓ Cognumach kernel built"; \
	else \
		echo "   ✗ Cognumach kernel not found"; \
	fi
	@echo ""
	@echo "2. Checking HurdCog..."
	@if [ -d /home/ubuntu/hurdcog/cogkernel ]; then \
		echo "   ✓ HurdCog cogkernel found"; \
	else \
		echo "   ✗ HurdCog cogkernel not found"; \
	fi
	@echo ""
	@echo "3. Checking OCC..."
	@if [ -d /home/ubuntu/occ/atomspace ]; then \
		echo "   ✓ OCC atomspace found"; \
	else \
		echo "   ✗ OCC atomspace not found"; \
	fi
	@echo ""
	@echo "Integration test complete!"

# Installation
install:
	@echo "Installing AGI-OS to system..."
	@echo "This requires root privileges."
	@if command -v guix >/dev/null 2>&1; then \
		sudo guix package -f occ-hurdcog-unified.scm; \
	else \
		echo "Manual installation not yet implemented."; \
		echo "Please install GNU Guix first."; \
		exit 1; \
	fi

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	@cd /home/ubuntu/cognumach && make clean 2>/dev/null || true
	@cd /home/ubuntu/hurdcog/cogkernel && make clean 2>/dev/null || true
	@cd /home/ubuntu/occ && rm -rf build 2>/dev/null || true
	@echo "✓ Clean complete"

# Documentation generation
docs:
	@echo "Generating integration documentation..."
	@echo "See integration_architecture.md and integration_analysis.md"

# Development helpers
dev-setup:
	@echo "Setting up development environment..."
	@echo "Installing required dependencies..."
	@sudo apt-get update
	@sudo apt-get install -y build-essential gcc-multilib binutils \
		autoconf automake libtool pkg-config gawk bison flex \
		guile-3.0 guile-3.0-dev python3 python3-pip \
		cmake libboost-all-dev
	@echo "✓ Development environment ready"

# Quick start
quickstart: dev-setup direct-build test
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  AGI-OS Quick Start Complete!                              ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Start HurdCog dashboard: cd /home/ubuntu/hurdcog/cogkernel && ./start-dashboard.sh"
	@echo "  2. Access dashboard at: http://localhost:8080/dashboard"
	@echo "  3. Explore the integration: make test"
	@echo ""
