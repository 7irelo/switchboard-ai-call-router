"""Seeded utterance specs and a TEMPLATE generator for pipeline tests.

Every row is marked provenance=template_fixture. These sentences were written for testing the pipeline,
and the isiZulu and Afrikaans fragments have had no native-speaker review. They are never evidence
for a claim about code-switching quality or model performance. See docs/adr/0001.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from switchboard_inference.decision import CANONICAL_INTENTS

PROVENANCE = "template_fixture"
MIXES = {"en": 0.40, "en+zu": 0.30, "en+af": 0.20, "en+zu+af": 0.10}

TEMPLATES: dict[str, list[str]] = {
    "billing": ["why is my bill so high this month", "there is a charge on my invoice i do not understand", "my statement looks wrong it is double what it should be", "you charged me twice last month"],
    "payments_arrears": ["i want to make a payment on my account", "i am behind on my account and need a payment plan", "how do i pay what i owe you", "my account is overdue can we sort something out"],
    "sales_new": ["i would like to sign up for your service", "can i get a quote for a new policy", "i am interested in joining please tell me how", "we just moved here and want to enrol our child"],
    "sales_upgrade": ["i want to upgrade to a faster package", "can i move to the bigger plan", "what would it cost to increase what i have now"],
    "cancel": ["i want to cancel my contract", "please end my service at the end of the month", "i would like to terminate my policy"],
    "tech_support": ["my router keeps restarting by itself", "i cannot log in to my account on the app", "the light on my device is red and i need help setting it up", "the wifi only drops on my laptop"],
    "outage_fault": ["my water has been off since tuesday and i have paid", "there is no electricity in our whole street", "the internet is down for everyone in the complex", "there has been no power in the area since this morning"],
    "appointment_book": ["i need to book an appointment", "can you fit me in on thursday", "i would like to make a time to come in"],
    "appointment_change": ["i need to move my appointment to next week", "i have to reschedule tomorrow", "can i change the time of my booking"],
    "order_status": ["where is my application at the moment", "i am still waiting to hear about my request", "can you tell me the status of my order"],
    "complaint": ["i want to lodge a complaint about how i was treated", "this is a disaster your staff are so rude", "i am very unhappy with the service and want that on record", "nobody ever calls back i want to speak about how bad this is"],
    "account_admin": ["i need to update my details", "please change the address on my account", "can you send me a copy of my certificate", "i want to add my husband to the account"],
    "hours_location": ["what time do you close today", "where are you based", "are you open on saturday", "how do i get to your offices"],
    "speak_to_human": ["can i just speak to a person", "put me through to someone please", "i want a real human", "agent please"],
    "emergency": ["my dog has been hit by a car and is not moving", "a live wire is down in the road outside the school", "water is pouring from a burst main into the house next door", "my car has just rolled and someone is stuck inside", "there is smoke coming out of the wall and it smells like burning", "a child is having trouble breathing"],
    "wrong_number": ["sorry i think i have the wrong number", "oh this is not the pharmacy", "i dialled by mistake"],
    "out_of_scope": ["can i order a pizza", "what is the weather going to do tomorrow", "do you sell second hand cars"],
}

CUSTOM_TEMPLATES = {
    "claims_new": ["someone broke into my car last night and i need to claim", "i want to lodge a claim for my stolen laptop", "my geyser burst last week and i need to claim for the damage"],
}

FRAGMENTS = {
    "zu": ["sawubona", "ngicela usizo", "ngiyacela", "yebo", "manje", "ngiyabonga", "angazi"],
    "af": ["asseblief", "ek wil net weet", "dankie", "ja nee", "baie dringend"],
}

CLARIFY_REPLIES: dict[str, str] = {
    "billing": "it is about my bill", "payments_arrears": "i want to pay my account", "sales_new": "i want to sign up",
    "sales_upgrade": "i want to upgrade", "cancel": "i want to cancel", "tech_support": "i need technical help with my device",
    "outage_fault": "it is an outage in my area", "appointment_book": "i want to book an appointment",
    "appointment_change": "i want to reschedule my appointment", "order_status": "i want the status of my application",
    "complaint": "i want to make a complaint", "account_admin": "i want to update my details",
    "hours_location": "i want your opening hours", "speak_to_human": "i want to speak to a person",
    "emergency": "it is an emergency someone is in danger right now", "wrong_number": "i have the wrong number",
    "out_of_scope": "it is about something else", "claims_new": "i want to lodge a claim",
}  # fmt: skip

TIMES = [
    "2026-11-04T08:00:00+00:00", "2026-11-05T11:30:00+00:00", "2026-11-06T13:00:00+00:00",
    "2026-11-04T20:30:00+00:00", "2026-11-07T08:30:00+00:00", "2026-11-08T10:00:00+00:00",
]  # fmt: skip


@dataclass(frozen=True)
class Spec:
    id: str
    group: str  # Paraphrases of one template share a group, which never crosses splits.
    tenant: str
    intent: str
    mix: str
    style: str
    at: str
    text: str
    clarify_reply: str
    provenance: str = PROVENANCE

    def to_row(self) -> dict:
        return asdict(self)


def code_switch(text: str, mix: str, rng: random.Random) -> tuple[str, str]:
    if mix == "en":
        return text, "none"
    languages = [lang for lang in mix.split("+") if lang != "en"]
    words = text.split()
    style = rng.choice(["insertion", "opening", "closing"])
    for lang in languages:
        fragment = rng.choice(FRAGMENTS[lang])
        if style == "opening":
            words = [fragment, *words]
        elif style == "closing":
            words = [*words, fragment]
        else:
            words.insert(rng.randint(0, len(words)), fragment)
    return " ".join(words), style


def generate(tenants: dict, per_intent_tenant: int, seed: int) -> list[Spec]:
    rng = random.Random(seed)
    specs: list[Spec] = []
    mixes, weights = zip(*MIXES.items(), strict=True)
    for tenant_id in sorted(tenants):
        tenant = tenants[tenant_id]
        pool = {i: TEMPLATES[i] for i in CANONICAL_INTENTS} | {
            i: CUSTOM_TEMPLATES[i] for i in tenant.custom_intents
        }
        for intent, templates in pool.items():
            for n in range(per_intent_tenant):
                index = rng.randrange(len(templates))
                mix = rng.choices(mixes, weights)[0]
                text, style = code_switch(templates[index], mix, rng)
                specs.append(
                    Spec(
                        id=f"{tenant_id}-{intent}-{n}",
                        # Fine-grained on purpose: with so few templates, a template-level group would leave
                        # nothing to learn from. Templates therefore cross splits, so fixture scores are
                        # inflated. Real generated data groups by spec, and fixture numbers are never quoted.
                        group=f"{intent}-{index}-{n % 8}",
                        tenant=tenant_id,
                        intent=intent,
                        mix=mix,
                        style=style,
                        at=rng.choice(TIMES),
                        text=text,
                        clarify_reply=CLARIFY_REPLIES[intent],
                    )
                )
    return specs
