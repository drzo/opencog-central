;
; remote-query.scm
;
; Demo of using queries to obtain sharply-defined sets of Atoms
; from the remote server. These queries generalize the idea of
; fetching the incoming set, or fetching the subset of Atoms of
; some given type, by allowing a highly-detailed search pattern
; to be sent to the remote server, to get exactly those atoms that
; you want.
;
; There are three generic types of queries: MeetLinks, JoinLinks
; and QueryLinks.
; * MeetLinks resemble the idea of a "meet" from set theory (or
;   lattice theory): they effectively say "get me all Atoms that
;   satisfy properties X and Y and Z". The wiki, and atomspace
;   tutorials provide much, much more information.
; * JoinLinks resemble the idea of a "join" from set theory (or
;   lattice theory): they effectively say "get me everything that
;   contains Atom X". Again, the wiki and other examples explain
;   more.
; * QueryLinks are rewrite rules: they perform a Meet, and then
;   take those results to create some new structure.
;
; Each of these query types can be used to fetch Atoms from the
; remote server.
;
; -------------------------------
; Basic initialization and set-up
;
(use-modules (opencog) (opencog persist))
(use-modules (opencog persist-cog))
(cogserver-open "cog://localhost")

; Or, instead, for the simple client, say
; (use-modules (opencog persist-cog-simple))
; (cog-simple-open "cog://localhost")

; -----------------------
; Populate the Atomspace.
;
; The demo needs to have some Atoms over at the remote server
; so that they can be fetched. Set that up here.
(List (Concept "A") (Concept "B"))
(Set (Concept "A") (Concept "B"))
(Set (Concept "A") (Concept "B") (Concept "C"))
(Evaluation (Predicate "foo")
	(List (Concept "B") (Concept "C") (Concept "oh boy!")))

; Push the entire atomspace out to the remote server.
(store-atomspace)

; Clear the local AtomSpace.
(cog-atomspace-clear)

; Verify that the current AtomSpace is indeed empty.
(cog-get-all-roots)

; -------------------------
; Querying with Meet links.
;
; The (List (Concept "A") (Concept "B")) can be thought of as a directed
; arrow from head to tail. Write a query, that, given the head finds the
; tail.
(define get-tail (Meet (List (Concept "A") (Variable "tail"))))

; Define a key where the results will be placed. This allows the query
; to run asynchronously; the results will be located at the key when
; they are finally available.
(define results-key (Predicate "results"))

; Find and fetch all tails at the remote server.
(fetch-query get-tail results-key)

; Take a look at what was found.
(cog-value get-tail results-key)

; -------------
; Query caching
;
; By default, the results of the query are cached at the server end.
; This is because queries can be CPU-intensive, and it's pointless to
; keep running them over and over. Of course, this can result in stale
; data. Try it ...
;
; Add some more data, push it out to the server, and delete it locally.
(List (Concept "A") (Concept "F"))
(store-atomspace)
(cog-extract-recursive! (Concept "F"))

; Verify that (Concept "F") is gone
(cog-get-all-roots)

; Re-run the query
(fetch-query get-tail results-key)

; Take a look at what was found. ... oh no, its the old cached result!
(cog-value get-tail results-key)

; There are two ways of handling this. One is to brute-force kill
; the cache. The can be done as so:
(cog-set-value! get-tail results-key #f)

; Now brute-force kill it in the server:
(store-value get-tail results-key)

; ... and rerun the query. This time, we expect all the results.
(fetch-query get-tail results-key)
(cog-value get-tail results-key)

; --------------
; Query metadata
;
; Caches have a basic problem: one does not know if they are expired,
; or fresh, or quite what is going on with them.  So its safe to always
; clobber the cache ... but that defeats the whole purpose of caching.
; One way to address this issue is to provide meta-data about the cache.
; The most important meta-data is the age of the cache. It can be gotten
; as shown below.
;
; query-cache metadata is currently a highly-experimental, and subject to
; change. The format of the meta-data in particular is not yet fixed.
; Don't be surprised if this part of the demo works strangely.
;
; Let's repeat some of the above.
(List (Concept "A") (Concept "G"))
(store-atomspace)
(cog-extract-recursive! (Concept "G"))

; Oh no! The cache is stale! Missing (Concept "G")!
(fetch-query get-tail results-key)
(cog-value get-tail results-key)

; We want meta-data, and we want a fresh re-computation.
(define metadata (Predicate "my metadata"))
(fetch-query get-tail results-key metadata #t)

; Yay! it worked!
(cog-value get-tail results-key)

; Tell us more!
(fetch-value get-tail metadata)
(cog-value get-tail metadata)

; Currently, the above returns seconds since January 1, 1970
(cog-value->list (cog-value get-tail metadata))

; Print the time-string... a bit verbose, but whatever.
(strftime "%c" (localtime  (inexact->exact (car
	(cog-value->list (cog-value get-tail metadata))))))

; Again, the format of the metadata is subject to change.

; ------------------------
; Generalized Incoming Set
;
; One of the most important queries is to fetch every graph containing
; a given Atom. This can be done with the JoinLink. The below fetches
; the entire incoming set of (Concept "B").  More info about JoinLinks
; can be found on the wiki page https://wiki.opencog.org/w/JoinLink

(define b-holders (MaximalJoin (Present (Concept "B"))))

: Just like before...
(fetch-query b-holders results-key)
(cog-value b-holders results-key)

; Verify that everything landed in the AtomSpace.
(cog-get-all-roots)

; That's all! Thanks for paying attention!
; ----------------------------------------
