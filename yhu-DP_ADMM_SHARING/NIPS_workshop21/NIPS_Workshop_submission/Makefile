#Makefile for LaTeX document files
# Created: July 3, 2015. Last modified: Feb 4, 2019.

# Set up the paths and names
# Default output is named "main". Default section path "./sections". Default figure path "./figures"
DOCUMENT=neurips_2020
SECPATH=sections/
FIGPATH=figures/

SECTIONS=$(wildcard $(SECPATH)*.tex)
FIGURES=$(wildcard $(FIGPATH)*.pdf)

$(DOCUMENT).pdf : $(DOCUMENT).tex $(SECTIONS) $(FIGURES) $(DOCUMENT).aux 
	pdflatex $(DOCUMENT).tex

$(FIGPATH)%.pdf : $(FIGPATH)%.eps 
	epstopdf $<

$(DOCUMENT).aux : $(DOCUMENT).bib
	pdflatex $(DOCUMENT).tex
	bibtex $(DOCUMENT)
	pdflatex $(DOCUMENT).tex
	pdflatex $(DOCUMENT).tex

all : 
	pdflatex $(DOCUMENT).tex
	bibtex $(DOCUMENT)
	pdflatex $(DOCUMENT).tex
	pdflatex $(DOCUMENT).tex
	pdflatex $(DOCUMENT).tex

clean:
	rm -rf *.aux $(DOCUMENT).log $(DOCUMENT).toc $(DOCUMENT).dvi \
    $(DOCUMENT).pdf $(DOCUMENT).bbl $(DOCUMENT).blg $(FIGPATH)*.pdf \
    $(DOCUMENT).out

