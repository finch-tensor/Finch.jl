DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io

OUTPUTS = \
    Looplets-lookup.png \
    Looplets-run.png \
    Looplets-spike.png \
    Looplets-sequence.png \
    Looplets-stepper.png \
    Looplets-phase.png \
    Looplets-switch.png \
    Looplets-thunk.png \
    LevelsVsFibers-tensor.png \
    LevelsVsFibers-matrix.png \
    Structures-structures.png \
    Structures-examples.png \
    Structures-levels.png \
	icon.png \
	logo.png \
	favicon.ico \

all: $(OUTPUTS)

Looplets-lookup.png: Looplets.drawio
	$(DRAWIO) --export --page-index 0 --output "$@" "$<"

Looplets-run.png: Looplets.drawio
	$(DRAWIO) --export --page-index 1 --output "$@" "$<"

Looplets-spike.png: Looplets.drawio
	$(DRAWIO) --export --page-index 2 --output "$@" "$<"

Looplets-sequence.png: Looplets.drawio
	$(DRAWIO) --export --page-index 3 --output "$@" "$<"

Looplets-stepper.png: Looplets.drawio
	$(DRAWIO) --export --page-index 4 --output "$@" "$<"

Looplets-phase.png: Looplets.drawio
	$(DRAWIO) --export --page-index 5 --output "$@" "$<"

Looplets-switch.png: Looplets.drawio
	$(DRAWIO) --export --page-index 6 --output "$@" "$<"

Looplets-thunk.png: Looplets.drawio
	$(DRAWIO) --export --page-index 7 --output "$@" "$<"

LevelsVsFibers-tensor.png: LevelsVsFibers.drawio
	$(DRAWIO) --export --page-index 0 --output "$@" "$<"

LevelsVsFibers-matrix.png: LevelsVsFibers.drawio
	$(DRAWIO) --export --page-index 1 --output "$@" "$<"

Structures-structures.png: Structures.drawio
	$(DRAWIO) --export --page-index 0 --output "$@" "$<"

Structures-examples.png: Structures.drawio
	$(DRAWIO) --export --page-index 1 --output "$@" "$<"

Structures-levels.png: Structures.drawio
	$(DRAWIO) --export --page-index 2 --output "$@" "$<"

logo.png: logo.drawio
	$(DRAWIO) --export --page-index 0 --scale=2.0 --output "$@" "$<"

icon.png: logo.drawio
	$(DRAWIO) --export --page-index 0 --scale=0.5 --output "$@" "$<"

favicon.ico: icon.png # requires ImageMagick
	magick -background transparent $^ -define icon:auto-resize=16,24,32,48,64,72,96,128,256 $@

clean:  ## Clean output files
	rm -f $(OUTPUTS)