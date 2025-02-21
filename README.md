# burst_analyzer
This is an interactive tool to read in burst snippet in filterbank format, zap RFI, and calculate burst properties. 

# Usage
python CLASSY_burst_analyzer.py -b burst.fil -d 123 -t NRT -v

Extracted burst properties:
- Fluence (Jy ms), calculated for the selected event duration over the entire bandwidth
- Peak flux density (Jy)
- Event duration (ms), manually selected 
- Spectral extent (MHz), manually selected
- MJD: Topocentric at NRT