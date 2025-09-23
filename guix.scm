;;; Top-level Guix package file for OpenCog Collection
;;; Use with: guix build -f guix.scm

(use-modules (guix packages)
             (guix download)
             (guix git-download)
             (guix gexp)
             (guix build-system python)
             (guix build-system cargo)
             (guix build-system cmake)
             (guix build-system gnu)
             (guix build-system trivial)
             (guix build utils)
             (guix build cmake-build-system)
             ((guix licenses) #:prefix license:)
             (gnu packages)
             (gnu packages python)
             (gnu packages python-xyz)
             (gnu packages python-science)
             (gnu packages machine-learning)
             (gnu packages rust)
             (gnu packages crates-io)
             (gnu packages cmake)
             (gnu packages pkg-config)
             (gnu packages guile)
             (gnu packages boost)
             (gnu packages serialization)
             (gnu packages databases)
             (gnu packages version-control)
             (gnu packages maths)
             (gnu packages cpp)
             (gnu packages check)
             (gnu packages build-tools))

(define-public opencog-collection
  (package
    (name "opencog-collection")
    (version "0.1.0")
    (source (local-file "." "opencog-collection-checkout"
                        #:recursive? #t))
    (build-system cmake-build-system)
    (arguments
      `(#:tests? #f  ; Disable tests for now as they may require network access
        #:configure-flags
        ,(list "-DCMAKE_BUILD_TYPE=Release"
               "-DBUILD_COGUTIL=ON"
               "-DBUILD_ATOMSPACE=ON"
               "-DBUILD_COGSERVER=ON"
               "-DBUILD_MATRIX=ON"
               "-DBUILD_LEARN=ON"
               "-DBUILD_AGENTS=ON"
               "-DBUILD_SENSORY=ON"
               "-DBUILD_ATOMSPACE_STORAGE=OFF"  ; Disable storage for now to reduce complexity
               "-DBUILD_ATOMSPACE_EXTENSIONS=OFF"
               "-DCMAKE_INSTALL_PREFIX=/gnu/store")
        #:phases
        ,(modify-phases %standard-phases
           (add-before 'configure 'check-dependencies
             (lambda* (#:key inputs outputs #:allow-other-keys)
               ;; Check that required subdirectories exist
               (for-each (lambda (dir)
                          (unless (file-exists? dir)
                            (format #t "Warning: Directory ~a not found~%" dir)))
                        '("cogutil" "atomspace" "cogserver" "matrix" "learn" "agents" "sensory"))
               #t))
           (add-before 'configure 'set-environment
             (lambda* (#:key inputs outputs #:allow-other-keys)
               ;; Set up environment for building
               (when (assoc-ref inputs "boost")
                 (setenv "BOOST_ROOT" (assoc-ref inputs "boost")))
               (when (assoc-ref inputs "pkg-config")
                 (setenv "PKG_CONFIG_PATH"
                         (string-append (assoc-ref inputs "pkg-config") "/lib/pkgconfig:"
                                      (or (getenv "PKG_CONFIG_PATH") ""))))
               #t))
           (replace 'configure
             (lambda* (#:key configure-flags #:allow-other-keys)
               ;; Only configure if we have the required directories
               (if (and (file-exists? "cogutil")
                       (file-exists? "atomspace"))
                   (begin
                     (mkdir-p "build")
                     (chdir "build")
                     (apply invoke "cmake" ".." configure-flags))
                   (begin
                     (format #t "Skipping CMake build - required directories not found~%")
                     #t))))
           (replace 'build
             (lambda* (#:key #:allow-other-keys)
               ;; Only build if we successfully configured
               (if (file-exists? "Makefile")
                   (invoke "make" "-j" (number->string (parallel-job-count)))
                   (begin
                     (format #t "Skipping make build - no Makefile found~%")
                     #t))))
           (replace 'install
             (lambda* (#:key outputs #:allow-other-keys)
               (let* ((out (assoc-ref outputs "out"))
                      (bin (string-append out "/bin"))
                      (lib (string-append out "/lib"))
                      (share (string-append out "/share/opencog-collection")))
                 (mkdir-p share)
                 (mkdir-p bin)
                 (mkdir-p lib)
                 
                 ;; Install CMake build if it exists
                 (when (file-exists? "Makefile")
                   (invoke "make" "install"))
                 
                 ;; Install Python components
                 (chdir "..")  ; Go back to source directory
                 (when (file-exists? "app.py")
                   (install-file "app.py" share)
                   (call-with-output-file (string-append bin "/opencog-demo")
                     (lambda (port)
                       (format port "#!/bin/sh~%exec python3 ~a/app.py \"$@\"~%"
                               share)))
                   (chmod (string-append bin "/opencog-demo") #o755))
                 
                 ;; Install Rust components if Cargo.toml exists
                 (when (file-exists? "Cargo.toml")
                   (setenv "CARGO_HOME" (string-append (getcwd) "/.cargo"))
                   (invoke "cargo" "build" "--release")
                   (when (file-exists? "target/release/hyperon")
                     (install-file "target/release/hyperon" bin))
                   (when (file-exists? "target/release/libhyperon.so")
                     (install-file "target/release/libhyperon.so" lib)))
                 
                 ;; Install documentation and metadata
                 (when (file-exists? "README.md")
                   (install-file "README.md" share))
                 (when (file-exists? "requirements.txt")
                   (install-file "requirements.txt" share))
                 
                 #t))))))
    (native-inputs
     (list pkg-config
           cmake
           rust
           `(,rust "cargo")))
    (inputs
     (list python
           python-numpy
           python-pandas
           python-scikit-learn
           python-matplotlib
           guile-3.0
           boost))
    (propagated-inputs
     (list python-numpy
           python-pandas
           python-scikit-learn
           python-matplotlib))
    (home-page "https://github.com/rzonedevops/occ")
    (synopsis "OpenCog Collection - Machine Learning Integration Environment")
    (description
     "This package provides the OpenCog Collection monorepo - an integrated
development environment for cognitive computing and artificial general intelligence (AGI).
The collection brings together multiple OpenCog-related projects into a coherent
whole for cognitive synergy.

The package includes the core OpenCog components when available:
@itemize
@item CogUtil - Base utilities and configuration system
@item AtomSpace - Hypergraph database and query engine
@item CogServer - Networking and communication layer
@item Matrix - Sparse vector and graph processing
@item Learn - Symbolic learning algorithms
@item Agents - Interactive cognitive agents
@item Sensory - Dataflow system for external world interaction
@end itemize

Additionally includes:
@itemize
@item Python-based machine learning demonstration using scikit-learn
@item Rust-based Hyperon cognitive computing framework
@item Complete source for research and development
@item Development environment for cognitive computing applications
@end itemize")
    (license license:gpl3+)))

;; Return the package for building
opencog-collection
