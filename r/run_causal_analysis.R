args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript run_causal_analysis.R INPUT.csv OUTPUT_DIR")
}

input_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

required <- c("unit_id", "period", "treated", "relative_period", "outcome", "value")
panel <- read.csv(input_path, stringsAsFactors = FALSE)
missing <- setdiff(required, names(panel))
if (length(missing) > 0) {
  stop(paste("missing columns:", paste(missing, collapse = ", ")))
}

if (!requireNamespace("fixest", quietly = TRUE)) {
  stop("The locked R package 'fixest' is required")
}

all_results <- list()
did_results <- list()
for (outcome_name in unique(panel$outcome)) {
  current <- panel[panel$outcome == outcome_name, ]
  period_levels <- sort(unique(current$period))
  current$period_index <- match(current$period, period_levels)
  current$unit_numeric <- as.integer(as.factor(current$unit_id))
  treated_units <- unique(current$unit_id[current$treated == 1])
  cohort_lookup <- setNames(rep(0L, length(unique(current$unit_id))), unique(current$unit_id))
  for (unit in treated_units) {
    rows <- current[current$unit_id == unit & current$relative_period == 0, ]
    if (nrow(rows) > 0) {
      cohort_lookup[[unit]] <- rows$period_index[[1]]
    }
  }
  current$treatment_cohort <- as.integer(cohort_lookup[current$unit_id])
  current$period_factor <- as.factor(current$period)
  current$unit_factor <- as.factor(current$unit_id)
  fit <- fixest::feols(
    value ~ i(relative_period, treated, ref = -1) | unit_factor + period_factor,
    cluster = ~unit_factor,
    data = current
  )
  table <- as.data.frame(fixest::coeftable(fit))
  table$term <- rownames(table)
  table$outcome <- outcome_name
  rownames(table) <- NULL
  all_results[[outcome_name]] <- table

  if (requireNamespace("did", quietly = TRUE) &&
      any(current$treatment_cohort > 0) &&
      any(current$treatment_cohort == 0)) {
    did_fit <- did::att_gt(
      yname = "value",
      tname = "period_index",
      idname = "unit_numeric",
      gname = "treatment_cohort",
      data = current,
      panel = TRUE,
      control_group = "nevertreated",
      est_method = "dr",
      clustervars = "unit_numeric",
      print_details = FALSE
    )
    dynamic <- did::aggte(did_fit, type = "dynamic")
    did_results[[outcome_name]] <- data.frame(
      outcome = outcome_name,
      event_time = dynamic$egt,
      att = dynamic$att.egt,
      standard_error = dynamic$se.egt
    )
  }
}

result <- do.call(rbind, all_results)
write.csv(result, file.path(output_dir, "fixest_event_study.csv"), row.names = FALSE)
if (length(did_results) > 0) {
  write.csv(
    do.call(rbind, did_results),
    file.path(output_dir, "callaway_santanna_dynamic_att.csv"),
    row.names = FALSE
  )
}

metadata <- list(
  status = "complete",
  estimator = "Callaway-Sant'Anna doubly robust ATT plus fixest event study",
  note = paste(
    "Callaway-Sant'Anna did estimates require a cohort variable and never-treated controls;",
    "the Python readiness gate prevents unsupported causal claims when those are absent."
  )
)
jsonlite::write_json(metadata, file.path(output_dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE)
