;
; metta-basic-test.scm - Test some basic MeTTa syntax compatibility
; Test for the matching `metta-lisp.scm` example demo.
;
(use-modules (opencog) (opencog exec))
(use-modules (opencog metta))
(use-modules (opencog test-runner))

(opencog-test-runner)
(define tname "metta-lisp-test")
(test-begin tname)

; Define a factorial function
(MeTTa "(= (fact $x) (if (< $x 2) 1 (* $x (fact (- $x 1)))))")

; Run it.
(define fact5 (cog-execute! (MeTTa "(fact 5)")))

; Expect this
(define ef5 (MeTTa "120"))

(test-assert "factorial-five" (equal? fact5 ef5))

; Define a simple named numeric value
(MeTTa "(= foo 6)")

; Run it.
(define fact6 (cog-execute! (MeTTa "(fact foo)")))
; Expect this
(define ef6 (MeTTa "720"))

(test-assert "factorial-six" (equal? fact6 ef6))

(test-end tname)

(opencog-test-end)
