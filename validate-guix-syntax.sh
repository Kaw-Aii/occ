#!/bin/bash
# Validate Guix package definition syntax

echo "Validating guix.scm syntax..."

# Check if guix is available
if ! command -v guix &> /dev/null; then
    echo "Guix is not installed. Performing basic Scheme syntax check..."
    
    # Check for balanced parentheses
    if command -v guile &> /dev/null; then
        guile -c "(use-modules (ice-9 rdelim)) (with-input-from-file \"guix.scm\" (lambda () (read)))" 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Basic Scheme syntax appears valid"
        else
            echo "✗ Scheme syntax errors detected"
            exit 1
        fi
    else
        echo "Neither guix nor guile available for validation"
        echo "Checking for balanced parentheses manually..."
        
        # Simple parenthesis balance check
        python3 << 'PYEOF'
with open('guix.scm', 'r') as f:
    content = f.read()
    open_count = content.count('(')
    close_count = content.count(')')
    if open_count == close_count:
        print(f"✓ Parentheses balanced: {open_count} pairs")
    else:
        print(f"✗ Parentheses unbalanced: {open_count} open, {close_count} close")
        exit(1)
PYEOF
    fi
else
    echo "Guix is available, performing full validation..."
    guix build -f guix.scm --dry-run
fi

echo "Validation complete!"
