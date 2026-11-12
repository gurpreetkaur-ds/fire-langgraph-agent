# Payments Service — Incident Runbook

## Severity levels
Sev-1: customers cannot pay or payments are double-charged. Page the on-call engineer and the payments lead immediately; open a status-page incident within 15 minutes.
Sev-2: degraded performance, p95 checkout latency above 2 seconds for more than 10 minutes. Page the on-call engineer.
Sev-3: internal tooling issues with no customer impact. Create a ticket; fix during business hours.

## First response
1. Acknowledge the page within 5 minutes.
2. Check the "Payments Overview" dashboard and the error budget burn rate.
3. If a deploy happened in the last 60 minutes, roll it back first and investigate second.

## Database failover
The primary PostgreSQL cluster fails over automatically when health checks fail for 30 seconds. Manual failover requires approval from the database owner and is run with the `pg-failover` playbook.

## Post-incident
Every Sev-1 and Sev-2 needs a blameless post-mortem within 5 business days, with action items tracked in the RELIAB project.
