# Intent rubric

**Status: draft, written by the AI assistant.** A second human rater hasn't checked it (M3 gate). Label by what the caller needs to happen next, not the words they use.

| Intent | Use when the caller needs... |
|---|---|
| `billing` | an explanation or correction of a charge, statement or invoice |
| `payments_arrears` | to make a payment, arrange one, or deal with overdue amounts |
| `sales_new` | to buy or start something they don't have |
| `sales_upgrade` | to change an existing service to a bigger or better one |
| `cancel` | to end a service or contract |
| `tech_support` | help with a problem specific to them |
| `outage_fault` | to report or check a service failure that isn't specific to their setup |
| `appointment_book` | a new appointment |
| `appointment_change` | to move or cancel an existing appointment |
| `order_status` | to know where an order, application or request has got to |
| `complaint` | to register dissatisfaction, with no other action as the main goal |
| `account_admin` | to change details, access or documents on an account |
| `hours_location` | opening hours, address or directions |
| `speak_to_human` | a person, with no reason given |
| `emergency` | urgent help because of immediate danger to life, safety or property |
| `wrong_number` | nothing from this business: they reached it by mistake |
| `out_of_scope` | something this business doesn't do |

## Hard pairs

| Pair | Rule |
|---|---|
| `billing` / `payments_arrears` | "Why is it this much?" is billing. "How do I pay?" or "I'm behind" is arrears. |
| `tech_support` / `outage_fault` | Fault in the area or service: outage. Fault in their own device or setup: support. Unknown: outage only if they mention others or a whole area. |
| `complaint` / `emergency` | Words like "disaster" or "nightmare" don't make an emergency. Immediate danger does, even in a calm voice. |
| `complaint` / the fix | "I've called three times and my bill's still wrong" is billing: the caller wants it fixed. Choose complaint only when they want it on record or escalated. |
| `cancel` / `complaint` | A threat to leave inside a complaint is a complaint. A request to end is cancel. |
| `speak_to_human` / any | If they ask for a person and give a reason, label the reason. |
| Two intents in one turn | Label the first thing that must happen. Emergency always wins. |
