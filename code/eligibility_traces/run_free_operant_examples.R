#!/usr/bin/env Rscript

source("eligibility_test.R")
example_summary <- run_example_scenarios(show_plots = interactive())
print(example_summary)
