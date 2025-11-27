TEX=main

LATEX=pdflatex
BIBTEX=bibtex

FIGS=pit_cosmic_evolution_plot.png bet_plot.png breaking_point_heatmap.png habit_field_vectors.png
BIBS=mainNotes.bib references.bib

.PHONY: all clean arxiv

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(BIBTEX) $(TEX)
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(LATEX) -interaction=nonstopmode $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).toc

# Stage a submission bundle for arXiv.
arxiv: clean
	rm -rf arxiv arxiv.zip
	mkdir -p arxiv
	cp $(TEX).tex $(BIBS) $(FIGS) arxiv/
	cd arxiv && zip -rq ../arxiv.zip .
