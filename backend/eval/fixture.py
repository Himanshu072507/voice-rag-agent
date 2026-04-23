"""Self-contained, fictional handbook used as the eval fixture.

Facts are invented so the answer must come from context, not LLM world knowledge.
Keep this text stable — the golden Q&A pairs in golden.json reference it directly.
"""

HANDBOOK_TEXT = """NimbusCraft Co. — Employee Handbook (Excerpt)

1. Company Background
NimbusCraft Co. was founded in 2019 by Elena Marquez and Theo Park in Portland,
Oregon. The company builds custom weather instruments for amateur meteorologists
and hobbyist storm chasers.

2. Flagship Products
NimbusCraft sells three flagship products. The Stratus-7 is a home weather
station priced at $349 and won the 2022 OutdoorTech Innovation Award. The
Cirrus Pro is a portable anemometer priced at $129. The CloudDeck software
platform is a subscription service at $12 per month.

3. Work Schedule
Employees work Tuesday through Friday, from 10 AM to 6 PM Pacific Time.
Mondays are designated as "Maker Mondays" — unstructured personal R&D days
during which employees are encouraged to explore side projects.

4. Remote Policy
NimbusCraft is a fully remote company. Employees receive an optional
coworking stipend of $200 per month. All-hands meetings are held on the
second Thursday of each month.

5. Time Off
Paid time off is unlimited, with a mandatory minimum of 15 days per year.
Parental leave is 18 weeks, fully paid, for both primary and secondary
caregivers. Sabbaticals of up to 3 months are available to employees after
5 years of tenure.

6. Benefits
Health insurance premiums are 100% covered by the company, and dental
premiums are 80% covered. The company offers a 401(k) plan with a 6%
company match. Each employee receives an annual wellness stipend of $1,500.

7. Equipment
New hires receive a 14-inch MacBook Pro, a Herman Miller Aeron ergonomic
chair, and a Stratus-7 weather station at no cost.

8. Customer Base
NimbusCraft has over 47,000 active users across 23 countries. Its three
largest markets are the United States (62% of users), Germany (14%), and
Japan (9%).

9. Engineering Stack
The backend is written in Rust. The frontend is built with Svelte. Services
are deployed on self-hosted Kubernetes clusters located in Oregon and
Frankfurt.

10. Compliance
NimbusCraft has been SOC 2 Type II certified since 2021 and is GDPR
compliant. Company policy prohibits sharing user data with third-party
analytics providers without explicit user consent.
"""
