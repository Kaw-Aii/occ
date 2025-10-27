# Workflow Update Note - October 2025

## Issue with Workflow Push

The improved `guix-build.yml` workflow could not be pushed via GitHub App due to missing `workflows` permission. The workflow improvements are documented here for manual application.

## Workflow Improvements to Apply Manually

The following improvements should be manually applied to `.github/workflows/guix-build.yml`:

### 1. Add Document Start Marker
```yaml
---
name: Guix Build CI
```

### 2. Add Workflow Dispatch Trigger
```yaml
on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]
  workflow_dispatch:  # Add this line
```

### 3. Add Timeout to Job
```yaml
jobs:
  guix-build:
    runs-on: ubuntu-latest
    timeout-minutes: 120  # Add this line
```

### 4. Improve Checkout Step
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    submodules: recursive  # Add this line
```

### 5. Improve Guix Daemon Setup

Replace the "Setup Guix environment" step with:

```yaml
- name: Setup Guix environment and daemon
  run: |
    # Detect current user
    CURRENT_USER=$(whoami)
    echo "Current user: $CURRENT_USER"
    
    # Set PATH for Guix
    GUIX_PATH="/var/guix/profiles/per-user/${CURRENT_USER}/current-guix/bin"
    export PATH="${GUIX_PATH}:$PATH"
    
    # Add to GITHUB_PATH for subsequent steps
    echo "${GUIX_PATH}" >> $GITHUB_PATH
    
    # Start guix daemon
    echo "Starting Guix daemon..."
    if sudo systemctl start guix-daemon 2>/dev/null; then
      echo "Guix daemon started via systemctl"
    else
      echo "Starting Guix daemon manually..."
      sudo /var/guix/profiles/per-user/root/current-guix/bin/guix-daemon \
        --build-users-group=guixbuild &
      DAEMON_PID=$!
      echo "Guix daemon started with PID: $DAEMON_PID"
    fi
    
    # Wait for daemon to be ready
    echo "Waiting for Guix daemon to be ready..."
    for i in {1..30}; do
      if guix describe &>/dev/null; then
        echo "✓ Guix daemon is ready (attempt $i)"
        break
      fi
      echo "Waiting for daemon... ($i/30)"
      sleep 2
    done
    
    # Verify daemon is accessible
    if ! guix describe; then
      echo "ERROR: Guix daemon failed to start properly"
      exit 1
    fi
    
    echo "Guix daemon is healthy and ready"
```

### 6. Add Validation Step

Add before the build step:

```yaml
- name: Validate Guix package definition
  run: |
    echo "Validating guix.scm syntax..."
    
    # Run validation script
    bash validate-guix-syntax.sh
    
    # Perform dry-run to check dependencies
    echo "Performing dry-run build..."
    guix build -f guix.scm --dry-run --verbosity=1 || {
      echo "WARNING: Dry-run detected potential issues"
      echo "Continuing with actual build attempt..."
    }
```

### 7. Improve Build Step

Replace the "Build with Guix" step with:

```yaml
- name: Build OpenCog Collection with Guix
  run: |
    echo "Building OpenCog Collection..."
    echo "Build started at: $(date)"
    
    # Build with verbose output
    guix build -f guix.scm --verbosity=2 --keep-going || {
      echo "Build failed. Checking for partial results..."
      exit 1
    }
    
    echo "Build completed at: $(date)"
```

### 8. Add Build Results Display

Add after the build step:

```yaml
- name: Display build results
  if: success()
  run: |
    echo "Build successful!"
    
    # Show what was built
    BUILD_OUTPUT=$(guix build -f guix.scm 2>&1 | tail -1)
    echo "Build output: $BUILD_OUTPUT"
    
    if [ -d "$BUILD_OUTPUT" ]; then
      echo "Contents of build output:"
      ls -lah "$BUILD_OUTPUT"
    fi
```

### 9. Add Artifact Upload on Failure

Add at the end:

```yaml
- name: Upload build logs on failure
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: guix-build-logs
    path: |
      /var/log/guix/
    retention-days: 7
    if-no-files-found: warn
```

## Summary of Improvements

1. **PATH Persistence**: Uses `GITHUB_PATH` to persist PATH between steps
2. **Daemon Readiness**: Implements retry loop with timeout
3. **Error Handling**: Better error messages and validation
4. **Workflow Control**: Adds manual trigger and timeout
5. **Build Artifacts**: Uploads logs on failure for debugging
6. **Validation**: Dry-run before actual build

## How to Apply

These changes should be applied manually through the GitHub web interface or by a user with appropriate permissions. The improved workflow is available in the local repository at `.github/workflows/guix-build.yml` for reference.

## Alternative Approach

If direct workflow modification is not possible, consider:
1. Creating a pull request with these changes
2. Having a repository maintainer apply the changes
3. Using GitHub's workflow editor to make the changes

---

**Date**: October 27, 2025  
**Author**: Manus AI Agent
