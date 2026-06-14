# Paper Makefile — dual support for ctexart (M6) and cjc (M7)
MAIN = main
CJC_MAIN = cjc-main
LATEXMK = latexmk -xelatex

.PHONY: main cjc clean cleanall

# M6: original ctexart paper
main: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib FORCE_MAKE
	$(LATEXMK) $<

# M7: CJC template paper
cjc: $(CJC_MAIN).pdf

$(CJC_MAIN).pdf: $(CJC_MAIN).tex cjc.cls cjc.bst cjc-refs.bib FORCE_MAKE
	$(LATEXMK) $<

clean:
	$(LATEXMK) -c $(MAIN).tex
	$(LATEXMK) -c $(CJC_MAIN).tex

cleanall:
	$(LATEXMK) -C $(MAIN).tex
	$(LATEXMK) -C $(CJC_MAIN).tex

FORCE_MAKE:
