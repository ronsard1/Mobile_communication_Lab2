# Mobile Communications – Lab 2 (Unit 2 Mini Projects)

University of Rwanda, College of Science and Technology, School of ICT, Department of Computer Engineering
Group 3: Ronsard RUGIRANGOGA (223026673), Ivan MULISA (223026695), Dieudonne BYIRINGIRO (223026513)

Python simulations and tools for the three Unit 2 mini projects. No SDR hardware is needed.
The written report summarises the results; this repository holds the complete code, every chart
and every printed result, including the ones that are not reproduced in the report.

## Structure

```
Mini_Project_2_1_Friis_BER/
    mini_project_2_1.py      Friis path loss + 6 dB check; Monte Carlo BER of BPSK, QPSK, 16-QAM, 64-QAM;
                             rate-1/2 (7,5) convolutional code with soft Viterbi; log-distance model;
                             log-normal shadowing; throughput proxy against Shannon-Hartley
    results_2_1.txt          all printed numbers (BER tables, gaps, coding gain, ranges)
    figures/
Mini_Project_2_2_OFDM_CP/
    mini_project_2_2.py      16-QAM OFDM (N = 64), cyclic prefix, 3-tap multipath channel,
                             CP sweep 0-16 samples, constellations, waveform and channel plots
    results_2_2.txt
    figures/
Mini_Project_2_3_WPAN_WLAN_WMAN/
    tech_specs.csv           sourced dataset (sources S1-S8 listed in the script and the report)
    mini_project_2_3.py      derived metrics, trade-off charts, three scenario checks
    results_2_3.txt, results_2_3_dataset_with_metrics.csv
    figures/
```

## Run

Python 3.9 or later.

```
pip install -r requirements.txt
python Mini_Project_2_1_Friis_BER/mini_project_2_1.py          (about 40 s, 10^6 bits per BER point)
python Mini_Project_2_2_OFDM_CP/mini_project_2_2.py            (a few seconds)
python Mini_Project_2_3_WPAN_WLAN_WMAN/mini_project_2_3.py     (a few seconds)
```

Random seeds are fixed, so the numbers match the report. Each script prints its results, writes a
`results_*.txt` file and rewrites its `figures/` folder.
