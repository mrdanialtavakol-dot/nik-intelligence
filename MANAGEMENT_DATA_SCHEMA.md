# Management Data Schema — V0.8

## tasks
- task_id
- department
- title
- assignee_id
- created_by
- source
- priority
- status
- kpi_id
- created_at
- due_at
- followup_interval
- last_report_at
- next_followup_at
- escalation_level
- closed_at

## task_reports
- report_id
- task_id
- author_id
- submitted_at
- progress_pct
- blocker
- result
- evidence_url
- next_step

## audit_log
- event_id
- actor_id
- event_type
- entity_type
- entity_id
- before_json
- after_json
- created_at

## finance_transactions
- transaction_id
- external_id
- idempotency_key
- occurred_at
- source
- direction
- amount
- counterparty_id
- invoice_id
- debit_account
- credit_account
- classification_confidence
- bank_reconciled
- auto_post_eligible
- exception_reason
- posted_at

## sales_leads
- lead_id
- created_at
- source
- phone_hash / phone
- dedupe_key
- priority_score
- assigned_agent_id
- assigned_at
- first_touch_at
- first_touch_sla_met
- stage
- outcome
- loss_reason
- next_followup_at
- won_at
- revenue

## support
- ticket_id
- opened_at
- category
- priority
- owner_id
- first_response_at
- resolved_at
- sla_met
- escalated
- root_cause

## engineering
- work_item_id
- type (feature/bug/site/release)
- owner_id
- sprint
- status
- progress
- blocker
- eta
- release_version
- deployed_at

## department_reports
- department
- cadence
- responsible_user
- due_at
- submitted_at
- status
- executive_summary
- decisions_needed
