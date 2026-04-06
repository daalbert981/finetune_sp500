# Prompt Format for SP500 Executive Classification

This document describes the exact prompt setup used to fine-tune and query the model. Follow this format precisely to reproduce results.

## Message Structure

The model expects a chat-format input with three roles: **system**, **user**, and **assistant**.

### System Prompt

The system prompt is identical for every request:

```
This assistant is trained to code executive ranks and roles along the following categories with 1 or 0.

Ranks:
- VP: 1 if Vice President (VP), 0 otherwise
- SVP: 1 if Senior Vice President (SVP), 0 otherwise
- EVP: 1 if Executive Vice President (EVP), 0 otherwise
- SEVP: 1 if Senior Executive Vice President (SEVP), 0 otherwise
- Director: 1 if Director, 0 otherwise
- Senior Director: 1 if Senior Director, 0 otherwise
- MD: 1 if Managing Director (MD), 0 otherwise
- SMD: 1 if Senior Managing Director (SMD), 0 otherwise
- SE: 1 if Senior Executive, 0 otherwise
- VC: 1 if Vice Chair (VC), 0 otherwise
- SVC: 1 if Senior Vice Chair (SVC), 0 otherwise
- President: 1 if President of the parent company, 0 when President of subsidiary or division but not parent company.

Roles:
- Board: 1 when role suggests person is a member of the board of directors, 0 otherwise
- CEO: 1 when Chief Executive Officer of parent company, 0 when Chief Executive Officer of a subsidiary but not parent company.
- CXO: 1 when C-Suite title, i.e., Chief X Officer, where X can be any type of designation, 0 otherwise. Chief Executive Officer of the parent company. Not Chief AND Officer, e.g., only officer of a function.
- Primary: 1 when responsible for primary activity of value chain, i.e., Supply Chain, Manufacturing, Operations, Marketing & Sales, Customer Service and alike, 0 when not a primary value chain activity.
- Support: 1 when responsible for a support activity of the value chain, i.e., Procurement, IT, HR, Management, Strategy, HR, Finance, Legal, R&D, Investor Relations, Technology, General Counsel and alike, 0 when not support activity of the value.
- BU: 1 when involved with an entity/distinct unit responsible for Product, Customer, or Geographical domain/unit; or role is about a subsidiary, 0 when responsibility is not for a specific product/customer/geography area but, for example, for the entire parent company.
```

### User Prompt Template

The user message follows this exact template:

```
In {year} the company '{company}' had an executive with the name {full_name}, whose official role title was: '{role_title}'.
```

Where:
- `{year}` — the fiscal year (e.g., `2015`)
- `{company}` — the company name in lowercase (e.g., `hsbc finance corp`)
- `{full_name}` — the executive's full name in lowercase (e.g., `kathryn madison`)
- `{role_title}` — the official role title in lowercase (e.g., `chief executive officer`)

### Expected Output Format

The model outputs exactly two XML-style tags with semicolon-delimited binary labels:

```
<rank>vp=0;svp=0;evp=0;sevp=0;dir=0;sdir=0;md=0;smd=0;se=0;vc=0;svc=0;president=0</rank>
<role>board=0;ceo=1;cxo=0;primary=0;support=0;bu=0</role>
```

Each label is either `0` or `1`. The labels always appear in the exact order shown above.

## Full Example

**User:**
```
In 2015 the company 'hsbc finance corp' had an executive with the name kathryn madison, whose official role title was: 'chief executive officer'.
```

**Assistant:**
```
<rank>vp=0;svp=0;evp=0;sevp=0;dir=0;sdir=0;md=0;smd=0;se=0;vc=0;svc=0;president=0</rank>
<role>board=0;ceo=1;cxo=0;primary=0;support=0;bu=0</role>
```

## Python Example

```python
messages = [
    {
        "role": "system",
        "content": "This assistant is trained to code executive ranks and roles along the following categories with 1 or 0.\n\nRanks:\n- VP: 1 if Vice President (VP), 0 otherwise\n- SVP: 1 if Senior Vice President (SVP), 0 otherwise\n- EVP: 1 if Executive Vice President (EVP), 0 otherwise\n- SEVP: 1 if Senior Executive Vice President (SEVP), 0 otherwise\n- Director: 1 if Director, 0 otherwise\n- Senior Director: 1 if Senior Director, 0 otherwise\n- MD: 1 if Managing Director (MD), 0 otherwise\n- SMD: 1 if Senior Managing Director (SMD), 0 otherwise\n- SE: 1 if Senior Executive, 0 otherwise\n- VC: 1 if Vice Chair (VC), 0 otherwise\n- SVC: 1 if Senior Vice Chair (SVC), 0 otherwise\n- President: 1 if President of the parent company, 0 when President of subsidiary or division but not parent company.\n\nRoles:\n- Board: 1 when role suggests person is a member of the board of directors, 0 otherwise\n- CEO: 1 when Chief Executive Officer of parent company, 0 when Chief Executive Officer of a subsidiary but not parent company.\n- CXO: 1 when C-Suite title, i.e., Chief X Officer, where X can be any type of designation, 0 otherwise. Chief Executive Officer of the parent company. Not Chief AND Officer, e.g., only officer of a function.\n- Primary: 1 when responsible for primary activity of value chain, i.e., Supply Chain, Manufacturing, Operations, Marketing & Sales, Customer Service and alike, 0 when not a primary value chain activity.\n- Support: 1 when responsible for a support activity of the value chain, i.e., Procurement, IT, HR, Management, Strategy, HR, Finance, Legal, R&D, Investor Relations, Technology, General Counsel and alike, 0 when not support activity of the value.\n- BU: 1 when involved with an entity/distinct unit responsible for Product, Customer, or Geographical domain/unit; or role is about a subsidiary, 0 when responsibility is not for a specific product/customer/geography area but, for example, for the entire parent company."
    },
    {
        "role": "user",
        "content": "In 2015 the company 'hsbc finance corp' had an executive with the name kathryn madison, whose official role title was: 'chief executive officer'."
    }
]

# Expected output:
# <rank>vp=0;svp=0;evp=0;sevp=0;dir=0;sdir=0;md=0;smd=0;se=0;vc=0;svc=0;president=0</rank>
# <role>board=0;ceo=1;cxo=0;primary=0;support=0;bu=0</role>
```

## Recommended: GBNF Grammar for Structured Output

When using this model with llama.cpp or llama-cpp-python, we recommend using a GBNF grammar to guarantee the output format is parseable. The grammar constrains the formatting tokens but does not affect the model's classification decisions — at each binary label, the model freely chooses 0 or 1 based on its learned probabilities.

```
root ::= rank-tag "\n" role-tag

rank-tag ::= "<rank>" rank-pairs "</rank>"
rank-pairs ::= "vp=" bit ";svp=" bit ";evp=" bit ";sevp=" bit ";dir=" bit ";sdir=" bit ";md=" bit ";smd=" bit ";se=" bit ";vc=" bit ";svc=" bit ";president=" bit

role-tag ::= "<role>" role-pairs "</role>"
role-pairs ::= "board=" bit ";ceo=" bit ";cxo=" bit ";primary=" bit ";support=" bit ";bu=" bit

bit ::= "0" | "1"
```

Without grammar, the model may produce semantically correct but differently formatted output (e.g., `rank: vp=1;svp=0;...` or one label per line), which requires a more flexible parser.

## Label Definitions Summary

| Label | Category | Meaning |
|-------|----------|---------|
| vp | Rank | Vice President |
| svp | Rank | Senior Vice President |
| evp | Rank | Executive Vice President |
| sevp | Rank | Senior Executive Vice President |
| dir | Rank | Director |
| sdir | Rank | Senior Director |
| md | Rank | Managing Director |
| smd | Rank | Senior Managing Director |
| se | Rank | Senior Executive |
| vc | Rank | Vice Chair |
| svc | Rank | Senior Vice Chair |
| president | Rank | President of parent company (not subsidiary) |
| board | Role | Board of Directors member |
| ceo | Role | CEO of parent company (not subsidiary) |
| cxo | Role | C-Suite (Chief X Officer) |
| primary | Role | Primary value chain activity |
| support | Role | Support value chain activity |
| bu | Role | Business unit / subsidiary / geographic domain |
