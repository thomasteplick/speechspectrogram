/*
Create the spectrogram of speech waveforms in WAV files.  Option to
display the time plot of the wavform.
*/

package main

import (
	"fmt"
	"html/template"
	"log"
	"math"
	"math/cmplx"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
)

const (
	addr               = "127.0.0.1:8080"                     // http server listen address
	fileSpectrogram    = "templates/spectrogramtemplate.html" // html for speech time or spectrogram plots
	patternSpectrogram = "/speechspectrogram"                 // http handler for speech spectrogram
	xlabels            = 11                                   // # labels on x axis
	ylabels            = 11                                   // # labels on y axis
	dataDir            = "data/"                              // directory for the audio wav files
	rows               = 300                                  // rows in canvas
	cols               = 300                                  // columns in canvas
	sampleRate         = 8000                                 // Hz or samples/sec
	maxSamples         = 2 * sampleRate                       // max audio wav samples = 2 sec * sampleRate
	twoPi              = 2.0 * math.Pi                        // 2Pi
	bitDepth           = 16                                   // audio wav encoder/decoder sample size
	ncolors            = 5                                    // number of grayscale colors in spectrogram
)

// Type to contain all the HTML template actions
type PlotT struct {
	Grid   []string // plotting grid
	Status string   // status of the plot
	Xlabel []string // x-axis labels
	Ylabel []string // y-axis labels
	Domain string   // plot time domain, spectrogram domain
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

type Bound struct {
	start, stop int // word boundaries in the audio waveform
}

// Primary data structure for holding the spectrogram state
type Spectrogram struct {
	plot       *PlotT         // data to be distributed in the HTML template
	Endpoints                 // embedded struct
	nsamples   int            // number of audio wav samples
	grayscale  map[int]string // grayscale for spectrogram
	wordWindow int            // message word window to accumulate audio level
	fftSize    int            // FFT size for spectrogram
}

// Window function type
type Window func(n int, m int) complex128

// global variables
var (
	tmplSpectrogram *template.Template
	winType         = []string{"Bartlett", "Welch", "Hamming", "Hanning", "Rectangle"}
)

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// init parses the html template files
func init() {
	tmplSpectrogram = template.Must(template.ParseFiles(fileSpectrogram))
}

// findEndpoints finds the minimum and maximum data values
func (ep *Endpoints) findEndpoints(input []float64) {
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	for _, y := range input {

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
}

// processTimeDomain plots the time domain data from audio wav file
func (spg *Spectrogram) processTimeDomain(filename string) error {

	var (
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	spg.plot.Grid = make([]string, rows*cols)
	spg.plot.Xlabel = make([]string, xlabels)
	spg.plot.Ylabel = make([]string, ylabels)

	// Open the audio wav file
	f, err := os.Open(filepath.Join(dataDir, filename))
	if err == nil {
		defer f.Close()
		dec := wav.NewDecoder(f)
		bufInt := audio.IntBuffer{
			Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
			Data:   make([]int, maxSamples), SourceBitDepth: bitDepth}
		n, err := dec.PCMBuffer(&bufInt)
		if err != nil {
			fmt.Printf("PCMBuffer error: %v\n", err)
			return fmt.Errorf("PCMBuffer error: %v", err.Error())
		}
		bufFlt := bufInt.AsFloatBuffer()
		//fmt.Printf("%s samples = %d\n", filename, n)
		spg.nsamples = n

		endpoints.findEndpoints(bufFlt.Data)
		// time starts at 0 and ends at #samples*sampling period
		endpoints.xmin = 0.0
		// #samples*sampling period, sampling period = 1/sampleRate
		endpoints.xmax = float64(spg.nsamples) / float64(sampleRate)

		// EP means endpoints
		lenEPx := endpoints.xmax - endpoints.xmin
		lenEPy := endpoints.ymax - endpoints.ymin
		prevTime := 0.0
		prevAmpl := bufFlt.Data[0]

		// Calculate scale factors for x and y
		xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
		yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

		// This previous cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-bufFlt.Data[0])*yscale + .5)
		col := int((0.0-endpoints.xmin)*xscale + .5)
		spg.plot.Grid[row*cols+col] = "online"

		// Store the amplitude in the plot Grid
		for n := 1; n < spg.nsamples; n++ {
			// Current time
			currTime := float64(n) / float64(sampleRate)

			// This current cell location (row,col) is on the line (visible)
			row := int((endpoints.ymax-bufFlt.Data[n])*yscale + .5)
			col := int((currTime-endpoints.xmin)*xscale + .5)
			spg.plot.Grid[row*cols+col] = "online"

			// Interpolate the points between previous point and current point;
			// draw a straight line between points.
			lenEdgeTime := math.Abs((currTime - prevTime))
			lenEdgeAmpl := math.Abs(bufFlt.Data[n] - prevAmpl)
			ncellsTime := int(float64(cols) * lenEdgeTime / lenEPx) // number of points to interpolate in x-dim
			ncellsAmpl := int(float64(rows) * lenEdgeAmpl / lenEPy) // number of points to interpolate in y-dim
			// Choose the biggest
			ncells := ncellsTime
			if ncellsAmpl > ncells {
				ncells = ncellsAmpl
			}

			stepTime := float64(currTime-prevTime) / float64(ncells)
			stepAmpl := float64(bufFlt.Data[n]-prevAmpl) / float64(ncells)

			// loop to draw the points
			interpTime := prevTime
			interpAmpl := prevAmpl
			for i := 0; i < ncells; i++ {
				row := int((endpoints.ymax-interpAmpl)*yscale + .5)
				col := int((interpTime-endpoints.xmin)*xscale + .5)
				// This cell location (row,col) is on the line (visible)
				spg.plot.Grid[row*cols+col] = "online"
				interpTime += stepTime
				interpAmpl += stepAmpl
			}

			// Update the previous point with the current point
			prevTime = currTime
			prevAmpl = bufFlt.Data[n]

		}

		// Set plot status if no errors
		if len(spg.plot.Status) == 0 {
			spg.plot.Status = fmt.Sprintf("file %s plotted from (%.3f,%.3f) to (%.3f,%.3f)",
				filename, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
		}

	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range spg.plot.Xlabel {
		spg.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range spg.plot.Ylabel {
		spg.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// Welch's Method and Bartlett's Method variation of the Periodogram
func (spg *Spectrogram) calculatePSD(audio []float64, PSD []float64, fftWindow string, fftSize int) (float64, float64, error) {

	N := fftSize
	m := N / 2

	// map of window functions
	window := make(map[string]Window, len(winType))
	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	w, ok := window[fftWindow]
	if !ok {
		fmt.Printf("Invalid FFT window type: %v\n", fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", fftWindow)
	}

	bufN := make([]complex128, N)

	for j := 0; j < len(audio); j++ {
		bufN[j] = complex(audio[j], 0)
	}

	// zero-pad the remaining samples
	for i := len(audio); i < N; i++ {
		bufN[i] = 0
	}

	// window the N samples with chosen window
	for k := 0; k < N; k++ {
		bufN[k] *= w(k, m)
	}

	// Perform N-point complex FFT and add squares to previous values in PSD
	fourierN := fft.FFT(bufN)
	x := cmplx.Abs(fourierN[0])
	PSD[0] = x * x
	psdMax := PSD[0]
	psdAvg := PSD[0]
	for j := 1; j < m; j++ {
		// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
		xj := cmplx.Abs(fourierN[j])
		xNj := cmplx.Abs(fourierN[N-j])
		PSD[j] = xj*xj + xNj*xNj
		if PSD[j] > psdMax {
			psdMax = PSD[j]
		}
		psdAvg += PSD[j]
	}

	return psdAvg / float64(m), psdMax, nil
}

// findWords finds the word boundaries in the audio waveform
func (spg *Spectrogram) findWords(data []float64) ([]Bound, error) {

	// prevent oscillation about threshold
	const hystersis = 0.8

	var (
		old    float64 = 0.0
		new    float64 = 0.0
		cur    int     = 0
		start  int     = 0
		stop   int     = 0
		sum    float64 = 0.0
		k      int     = 0
		j      int     = 0
		L      int     = spg.nsamples
		max    float64 = 0.0
		bounds []Bound = make([]Bound, 0)
		avg    float64 = 0.0
	)

	// Find the maximum and normalize the data
	for i := 0; i < L; i++ {
		new = math.Abs(data[i])
		avg += new
		if new > max {
			max = new
		}
	}
	avg /= float64(L)

	// The number of samples in the audio level integration window.
	// Determines when the word and message ends
	// Convert wordWindow to ms
	win := int(float64(spg.wordWindow) * .001 / (1.0 / float64(sampleRate)))
	// Minimum audio integration to determine when word begins and ends
	levelSum := float64(win) * avg
	buf := make([]float64, win)

	for k < L {
		for k < L {
			new = math.Abs(data[k])
			old = buf[cur]
			buf[cur] = new
			sum += (new - old)
			cur = (cur + 1) % win
			if k >= stop+win && sum > levelSum {
				start = k - win
				bounds = append(bounds, Bound{start: start})
				k++
				break
			}
			k++
		}

		for k < L {
			new = math.Abs(data[k])
			old = buf[cur]
			buf[cur] = new
			sum += (new - old)
			cur = (cur + 1) % win
			if k > start+win && sum < levelSum*hystersis {
				stop = k
				bounds[j].stop = stop
				k++
				break
			}
			k++
		}
		j++
	}

	return bounds, nil
}

// inBoundsSample checks if the sample is inside word boundaries
func (spg *Spectrogram) inBoundsSample(smpl int, bounds []Bound) bool {
	margin := spg.fftSize / 2
	for _, bound := range bounds {
		if smpl > (bound.start-margin) && smpl < (bound.stop-margin) {
			return true
		}
	}
	return false
}

// processSpectrogram creates a spectrogram of the speech waveform
func (spg *Spectrogram) processSpectrogram(filename, fftWindow string, wordsOnly bool, fftSize int) error {

	// get audio samples from audio wav file
	// open and read the audio wav file
	// create wav decoder, audio IntBuffer, convert IntBuffer to audio FloatBuffer
	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
		bounds    []Bound   // word boundaries in the audio
	)

	spg.plot.Grid = make([]string, rows*cols)
	spg.plot.Xlabel = make([]string, xlabels)
	spg.plot.Ylabel = make([]string, ylabels)

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, fftSize/2)

	// Open the audio wav file
	f, err := os.Open(filepath.Join(dataDir, filename))
	if err == nil {
		defer f.Close()
		dec := wav.NewDecoder(f)
		bufInt := audio.IntBuffer{
			Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
			Data:   make([]int, maxSamples), SourceBitDepth: bitDepth}
		n, err := dec.PCMBuffer(&bufInt)
		if err != nil {
			fmt.Printf("PCMBuffer error: %v\n", err)
			return fmt.Errorf("PCMBuffer error: %v", err.Error())
		}
		bufFlt := bufInt.AsFloatBuffer()
		//fmt.Printf("%s samples = %d\n", filename, n)
		spg.nsamples = n
		// x-axis is time or sample, y-axis is frequency
		endpoints.xmin = 0.0
		endpoints.xmax = float64(spg.nsamples)
		endpoints.ymin = 0.0
		endpoints.ymax = float64(fftSize / 2) // equivalent to Nyquist critical frequency

		// Calculate scale factors to convert physical units to screen units
		xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
		yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

		// number of cells to interpolate in time and frequency
		// round up so the cells in the plot grid are connected
		ncellst := int((math.Ceil(float64(cols) * float64(fftSize/2) / float64(spg.nsamples))))
		ncellsf := int(math.Ceil(float64(rows) / float64((fftSize / 2))))

		stepTime := float64((fftSize / 2) / ncellst)
		stepFreq := 1.0 / float64(ncellsf)

		// if wordsOnly, only do calculatePSD for samples inside the word boundaries to minimize
		// checking the spectrum of noise.  This would give a broad range of frequencies which
		// is not of interest.
		if wordsOnly {
			// loop over fltBuf and find the speech bounds
			bounds, err = spg.findWords(bufFlt.Data)
			if err != nil {
				fmt.Printf("findWords error: %v", err)
				return fmt.Errorf("findWords error: %s", err.Error())
			}
		}

		// for loop over samples, increment by fftSize/2, calculatePSD on the batch
		// Overlap by 50% due to non-rectangular window to avoid Gibbs phenomenon
		for smpl := 0; smpl < spg.nsamples; smpl += fftSize / 2 {
			if !wordsOnly || spg.inBoundsSample(smpl, bounds) {
				// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
				end := smpl + fftSize
				if end > spg.nsamples {
					end = spg.nsamples
				}
				_, psdMax, err := spg.calculatePSD(bufFlt.Data[smpl:end], PSD, fftWindow, fftSize)
				if err != nil {
					fmt.Printf("calculatePSD error: %v\n", err)
					return fmt.Errorf("calculatePSD error: %v", err.Error())
				}

				// for loop over the frequency bins in the PSD
				for bin := 0; bin < fftSize/2; bin++ {
					// find the grayscale color based on bin power
					// largest power is black, smallest power is white
					// shades of gray in-between black and white
					var gs string
					r := PSD[bin] / psdMax
					if r < .1 {
						gs = spg.grayscale[4]
					} else if r < .25 {
						gs = spg.grayscale[3]
					} else if r < .5 {
						gs = spg.grayscale[2]
					} else if r < .8 {
						gs = spg.grayscale[1]
					} else {
						gs = spg.grayscale[0]
					}

					// interpolate in time
					interpTime := float64(smpl)
					for nct := 0; nct < ncellst; nct++ {
						col := int((interpTime-endpoints.xmin)*xscale + .5)
						if col >= cols {
							col = cols - 1
						}
						// interpolate in frequency
						interpFreq := float64(bin)
						for ncf := 0; ncf < ncellsf; ncf++ {
							row := int((endpoints.ymax-interpFreq)*yscale + .5)
							if row < 0 {
								row = 0
							}
							// Store the color in the plot Grid
							spg.plot.Grid[row*cols+col] = gs
							interpFreq += stepFreq
						}
						interpTime += stepTime
					}
				}
			}
		}
	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / ((xlabels - 1) * sampleRate)
	x := endpoints.xmin / sampleRate
	// First label is empty for alignment purposes
	for i := range spg.plot.Xlabel {
		spg.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Apply the  sampling rate in Hz to the y-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.ymax

	// Construct y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	// First label is empty for alignment purposes
	for i := range spg.plot.Ylabel {
		spg.plot.Ylabel[i] = fmt.Sprintf("%.0f", y*sf)
		y += incr
	}

	return nil
}

// Create a spectrogram of the speech waveform
func handleSpectrogram(w http.ResponseWriter, r *http.Request) {
	var (
		plot       PlotT
		spg        *Spectrogram
		wordWindow int  = 0
		wordsOnly  bool = false
	)

	// Determine operation to perform on the speech waveform:  play, add, delete
	wordOp := r.FormValue("wordop")
	if wordOp == "play" {
		filename := r.FormValue("fileplaydelete")
		if len(filename) == 0 {
			fmt.Println("Enter filename for playing the word")
			plot.Status = "Enter filename for playing the word"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		audiowavdir := r.FormValue("audiowavdir")
		if len(audiowavdir) == 0 {
			fmt.Println("Enter directory for the audio wav file")
			plot.Status = "Enter director for the audio wav file"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Determine if time or spectrogram domain plot
		domain := r.FormValue("domain")
		if domain == "spectrogram" {
			plot.Domain = "Spectrogram (Hz/sec)"
		} else {
			plot.Domain = "Time Domain (sec)"
		}

		if domain == "time" {
			spg = &Spectrogram{plot: &plot}
			err := spg.processTimeDomain(filepath.Join(audiowavdir, filename))
			if err != nil {
				fmt.Printf("processTimeDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Time Domain of %s plotted.", filepath.Join(dataDir, audiowavdir, filename))
			// Spectrogram Domain
		} else {
			fftWindow := r.FormValue("fftwindow")

			txt := r.FormValue("fftsize")
			fftSize, err := strconv.Atoi(txt)
			if err != nil {
				fmt.Printf("fftsize int conversion error: %v\n", err)
				plot.Status = fmt.Sprintf("fftsize int conversion error: %s", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}

			if len(r.FormValue("wordsonly")) > 0 {
				wordsOnly = true
				txt := r.FormValue("wordwindow")
				if len(txt) == 0 {
					fmt.Println("Word window not defined for spectrogram  domain")
					plot.Status = "Word window not defined for spectrogram  domain"
					// Write to HTTP using template and grid
					if err := tmplSpectrogram.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
				wordWindow, err = strconv.Atoi(txt)
				if err != nil {
					fmt.Printf("Conversion to int for 'wordwindow' error: %v\n", err)
					plot.Status = "Conversion to int for 'window' error"
					// Write to HTTP using template and grid
					if err := tmplSpectrogram.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
			}

			spg = &Spectrogram{plot: &plot, wordWindow: wordWindow, fftSize: fftSize}
			spg.grayscale = make(map[int]string)
			for i := 0; i < ncolors; i++ {
				spg.grayscale[i] = fmt.Sprintf("gs%d", i)
			}

			err = spg.processSpectrogram(filepath.Join(audiowavdir, filename), fftWindow, wordsOnly, fftSize)
			if err != nil {
				fmt.Printf("processSpectrogram error: %v\n", err)
				plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status += fmt.Sprintf("Spectrogram of %s plotted.", filepath.Join(dataDir, audiowavdir, filename))
		}

		// Play the audio wav if fmedia is available in the PATH environment variable
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			cmd := exec.Command(fmedia, filepath.Join(dataDir, audiowavdir, filename))
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)

			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}
	} else if wordOp == "new" {
		spg = &Spectrogram{plot: &plot}
		filename := r.FormValue("filenew")
		if len(filename) == 0 {
			fmt.Println("Enter filename for the new word")
			plot.Status = "Enter filename for the new word"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			// filename includes the audiowav folder; eg.,  audiowavX/cat.wav, were X = 0, 1, 2, ...
			cmd := exec.Command(fmedia, "--record", "-o", filepath.Join(dataDir, filename), "--until=5",
				"--format=int16", "--channels=mono", "--rate=8000", "-y", "--start-dblevel=-50", "--stop-dblevel=-20;1")
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
				plot.Status = fmt.Sprintf("stdout, stderr error from running fmedia: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}
		// delete
	} else if wordOp == "delete" {
		spg = &Spectrogram{plot: &plot}
		filename := r.FormValue("fileplaydelete")
		if len(filename) == 0 {
			fmt.Println("Enter filename for deleting the word from the vocabulary")
			plot.Status = "Enter filename for deleting the word from the vocabulary"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		audiowavdir := r.FormValue("audiowavdir")
		if len(audiowavdir) == 0 {
			fmt.Println("Enter directory for the audio wav file")
			plot.Status = "Enter directory for the audio wav file"
			// Write to HTTP using template and grid
			if err := tmplSpectrogram.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		if filepath.Ext(filename) == ".wav" {
			if err := os.Remove(path.Join(dataDir, audiowavdir, filename)); err != nil {
				plot.Status = fmt.Sprintf("Remove %s error: %v", filename, err)
				// Write to HTTP using template and grid
				if err := tmplSpectrogram.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}
	} else {
		fmt.Println("Enter spectrogram parameters.")
		plot.Status = "Enter spectrogram parameters"
		// Write to HTTP using template and grid
		if err := tmplSpectrogram.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Execute data on HTML template
	if err := tmplSpectrogram.Execute(w, spg.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}

}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for creating spectrograms

	// Create HTTP handler for spectrogram generation
	http.HandleFunc(patternSpectrogram, handleSpectrogram)
	fmt.Printf("Speech Spectrogram Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
