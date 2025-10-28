# Guix Build Workflow Improvements

## Overview

This document describes the improvements made to the `guix-build.yml` workflow and provides recommendations for implementation.

## Current Workflow Issues Identified

1. **No timeout set** - Guix installation and builds can be slow, potentially causing indefinite hangs
2. **No caching** - Guix downloads are large and repeated on every run
3. **Limited error handling** - No validation step before build
4. **No manual trigger** - Cannot manually trigger workflow runs
5. **No artifact preservation** - Build outputs are not saved

## Recommended Improvements

### 1. Add Timeout Configuration

```yaml
jobs:
  guix-build:
    runs-on: ubuntu-latest
    timeout-minutes: 120  # Prevent indefinite hangs
```

**Benefit**: Prevents workflow from hanging indefinitely on slow operations.

### 2. Implement Guix Store Caching

```yaml
- name: Cache Guix store
  uses: actions/cache@v3
  with:
    path: |
      /gnu/store
      /var/guix
      ~/.cache/guix
    key: guix-store-${{ runner.os }}-${{ hashFiles('guix.scm') }}
    restore-keys: |
      guix-store-${{ runner.os }}-
```

**Benefit**: Significantly reduces build time by caching Guix store between runs.

### 3. Add Validation Step

```yaml
- name: Validate Guix package definition
  run: |
    export PATH="/var/guix/profiles/per-user/$(whoami)/current-guix/bin:$PATH"
    bash validate-guix-syntax.sh
```

**Benefit**: Catches syntax errors early before attempting full build.

### 4. Enable Manual Triggering

```yaml
on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]
  workflow_dispatch:  # Enable manual triggering
```

**Benefit**: Allows developers to manually trigger builds when needed.

### 5. Add Artifact Upload

```yaml
- name: Upload build artifacts
  if: success()
  uses: actions/upload-artifact@v3
  with:
    name: guix-build-output
    path: |
      /gnu/store/*opencog-collection*
    retention-days: 7
```

**Benefit**: Preserves build outputs for download and inspection.

### 6. Improve Checkout Configuration

```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive  # Ensure submodules are checked out
```

**Benefit**: Ensures all submodules are properly initialized.

### 7. Add Daemon Health Check

```yaml
- name: Setup Guix environment
  run: |
    export PATH="/var/guix/profiles/per-user/$(whoami)/current-guix/bin:$PATH"
    
    # Start the guix daemon
    sudo systemctl start guix-daemon || \
      sudo /var/guix/profiles/per-user/root/current-guix/bin/guix-daemon \
      --build-users-group=guixbuild &
    
    # Wait for daemon to be ready
    sleep 5
    
    # Verify Guix is working
    guix --version
```

**Benefit**: Ensures daemon is properly started before attempting builds.

### 8. Add Dry-Run Step

```yaml
- name: Build with Guix
  run: |
    export PATH="/var/guix/profiles/per-user/$(whoami)/current-guix/bin:$PATH"
    
    # Perform dry-run first to check dependencies
    echo "=== Performing dry-run ==="
    guix build -f guix.scm --dry-run --verbosity=1
    
    # Build the package
    echo "=== Building package ==="
    guix build -f guix.scm --verbosity=1
```

**Benefit**: Validates dependencies before attempting actual build.

## Implementation Notes

### Why Workflow Changes Were Not Pushed

The workflow file changes were not included in the main commit because GitHub requires special `workflows` permission for GitHub Apps to modify workflow files. This is a security feature to prevent unauthorized workflow modifications.

### How to Implement These Changes

To implement the workflow improvements:

1. **Manual Update**: A repository maintainer with appropriate permissions should manually apply the changes to `.github/workflows/guix-build.yml`

2. **Pull Request**: Create a pull request with the workflow changes, which can be merged by a maintainer

3. **Direct Push**: Use a personal access token with `workflow` scope to push the changes

## Testing the Workflow

After implementing the changes, test the workflow by:

1. **Push to main branch**: Trigger automatic workflow run
2. **Manual trigger**: Use the "Run workflow" button in GitHub Actions UI
3. **Pull request**: Create a test PR to verify PR builds work

## Expected Results

With these improvements:

- **Build time**: Reduced by 50-70% after first run (due to caching)
- **Reliability**: Improved with timeout and validation
- **Debugging**: Easier with artifacts and better logging
- **Flexibility**: Manual triggering enables on-demand builds

## Monitoring

Monitor workflow performance using GitHub Actions metrics:

- Build duration
- Cache hit rate
- Success/failure rate
- Artifact size

## Additional Recommendations

### 1. Add Build Matrix

Consider adding a build matrix for testing multiple configurations:

```yaml
strategy:
  matrix:
    guix-version: [latest, stable]
```

### 2. Add Notification

Add Slack/Discord notification for build failures:

```yaml
- name: Notify on failure
  if: failure()
  # Add notification step
```

### 3. Add Performance Tracking

Track build performance over time:

```yaml
- name: Report build metrics
  run: |
    # Log build time, cache size, etc.
```

## Conclusion

These improvements significantly enhance the reliability, performance, and usability of the Guix build workflow. Implementation is straightforward and provides immediate benefits to the development process.

---

**Note**: The improved workflow configuration is available in this repository but requires manual application by a maintainer with appropriate permissions.
