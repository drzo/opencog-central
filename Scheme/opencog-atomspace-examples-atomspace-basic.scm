;
; basic.scm -- Basic guile usage example!
;
; See `opencog/guile/README` or http://wiki.opencog.org/w/Scheme
; for additional documentation.
;
; The AtomSpace guile module is called `opencog`, and not `atomspace`,
; as you might expect. This is a historical artifact, meant to preserve
; backwards compatibility with the earliest versions of OpenCog. Thus,
; you will see the string "opencog" littering the code, everywhere.
; Wherever you see it, keep in mind that it refers to the AtomSpace.
; All OpenCog modules are always AtomSpace sub-modules. There is nothing
; in the OpenCog project that can work without the AtomSpace!
;
; If you have installed this git repo with `make install`, then start
; guile simply by saying `guile` at the bash prompt `$`.  Otherwise,
; you will need to do this:
;
;    $ guile -L opencog/scm -L build
;
; where `build` is where-ever you built this git repo.
;
; Another possibility: add paths to your `~/.guile` file. For example:
;
;   (add-to-load-path "/home/yourname/atomspace/opencog/scm")
;

; Load the base module for the AtomSpace. Yes, it is called "opencog".
; This is a historical artifact; see above.
(use-modules (opencog))

; Create a Node and place it in the default AtomSpace.
(ConceptNode "asdf")

; Short names also work:
(Concept "foo")

; The part inside the quotes can be any UTF-8 string. The ConceptNode
; is the name of one of several dozen different Node types. Nodes
; *always* carry a string label, like the above. All Node types are
; backed by a C++ class that "does stuff", when executed; some of these
; are more interesting than others. This will come in later demos.
; Some special examples include Nodes that hold numbers and URL's.

; Nodes exist as objects in the AtomSpace, so if you say
(Concept "foo")
; a second time it will be exactly the same Node as before: nothing was
; added to the AtomSpace, because (Concept "foo") is already in it.
; That is, Nodes are globally unique, and are identified by their node
; type, and the string name. See
;    https://wiki.opencog.org/w/Node
; for more details.

; -----------------
; Links are lists of Atoms (which can be Nodes or Links).
(ListLink (Concept "foo") (Concept "bar"))
(ListLink (Concept "foo") (ListLink (Concept "bar") (Concept "bar")))

; Just like Nodes, Links are globally unique: saying
(ListLink (Concept "foo") (Concept "bar"))
; a second time does nothing; only the first utterance placed it into
; the AtomSpace. Each utterance gives exactly the same Link.
;
; See
;    https://wiki.opencog.org/w/Link
;    https://wiki.opencog.org/w/Atom
; for more details.

; -----------------
; Atoms can be accessed in such a way that they are not created if
; they do not already exist. Access the above:
(cog-node 'ConceptNode "asdf")

; Access an atom that does not exist. This will return the empty list.
(cog-node 'ConceptNode "qwerty")
(cog-link 'ListLink (cog-node 'ConceptNode "Oh no!"))

; -----------------
; All Atoms are ordinary scheme objects, and so conventional scheme
; can be used to hold references to them:
(define f (Concept "foo"))
f
(format #t "Let us print it out: ~A\n" f)
(define fff (ListLink f f f))
fff
(format #t "Here is a bunch: ~A\n" fff)

; The single quote-mark in front of 'ConceptNode means that it is a
; scheme symbol, and not a string:
(symbol? 'foo)
(symbol? "bar")

; -----------------
; The ConceptNode is an Atom type. All Atom types are "types" in
; the mathematical sense of "Type Theory". More plainly, this is
; more-or-less the same thing as a "type" in ordinary programming
; languages. Atomese is a typed language. All Atom types have a
; matching C++ class that runs "under the covers". The Atom type
; inheritance hierarchy runs more-or-less the same as the C++ class
; hierarchy; there are some subtle exceptions.
;
; Get a list of all Atomese types:
(cog-get-types)

; For more info, see
;    https://wiki.opencog.org/w/Atom_types
;
; -----------------
; In the above, note that the type hierarchy begins with `Value` and
; not with `Atom`. This distinction is important: Atoms can be stored
; in the AtomSpace, Values cannot. Atoms are globally unique; there
; can only ever be one instance of (Concept "asdf"), and every such
; reference always refers to the same Atom. Because Atoms are held in
; the AtomSpace, they continue to exist even if all scheme references
; to them are dropped. The AtomSpace maintains a reference to prevent
; them from being deleted.
;
; By contrast, a new Value is created (instantiated) every time it
; is mentioned; Values are always destroyed (garbage-collected) when
; there are no more references to them.
(FloatValue 0 1 2 3.14159)

; As the above shows, most Values are vectors. This is a storage-space
; and access-speed optimization. Later examples show how elements in the
; vector can be accessed and manipulated.

; Values can be attached to Atoms, using a key (as key-value pairs).
; Effectively, all Atoms are *also* key-value pair databases. Later
; examples explore this idea more deeply.
(cog-set-value! (Concept "asdf") (Predicate "some key") (FloatValue 4 5 6))
(cog-value (Concept "asdf") (Predicate "some key"))

; -----------------
; The above set and get can be done in pure Atomese. We'll need to load
; one more module:
(use-modules (opencog exec))

(cog-execute! (ValueOf (Concept "asdf") (Predicate "some key")))
(cog-execute! (SetValue
	(Concept "asdf") (Predicate "some key")
	(Node "this is the new thing")))
(cog-execute! (ValueOf (Concept "asdf") (Predicate "some key")))

; Unlike the above `cog-set-value!`, one cannot legally say
;   (SetValue (Concept "asdf") (Predicate "some key") (FloatValue 4 5 6))
; because the FloatValue is NOT an Atom! Only Atoms may appear in here.
; This is not much of a limitation; later demos expand on how to work
; with these.
;
; For more info, see
;    https://wiki.opencog.org/w/FloatValue
;    https://wiki.opencog.org/w/StringValue
;    https://wiki.opencog.org/w/ValueOf
;    https://wiki.opencog.org/w/SetValue
;    https://wiki.opencog.org/w/Execution

; -----------------
; The guile REPL shell has built-in commands that assist with
; documentation. For example, the following will print a list of
; all functions having the string "cog" in them. Note the comma
; in front; the REPL-shell commands all start with a comma.
,apropos cog

; The above can be shortened to just ,a:
,a cog

; Get the documentation of a given function:
,describe cog-new-node
,describe cog-node

; The above can also be shortened:
,d cog-link

; Atoms are paired with links to the wiki:
,d ConceptNode

; The End.
; That's all, Folks!
; ------------------
