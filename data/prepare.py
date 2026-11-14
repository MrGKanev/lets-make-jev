"""Build data/{support,news}_{dev,test}.jsonl from public sets (sources/licences: README.md, NOTES.md)."""
import json

import pandas as pd

B = "https://huggingface.co/datasets/mteb/banking77/resolve/main/{}.jsonl"
C = "https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/{}-00000-of-00001.parquet"
OOS = 42  # CLINC label id for out-of-scope
SEED = 0

TEAMS = {  # banking77 intent -> team
    "card_issues": """card_arrival card_delivery_estimate card_linking card_not_working card_swallowed
        compromised_card contactless_not_working disposable_card_limits get_disposable_virtual_card
        get_physical_card getting_spare_card getting_virtual_card lost_or_stolen_card order_physical_card
        activate_my_card card_about_to_expire virtual_card_not_working pin_blocked change_pin
        apple_pay_or_google_pay atm_support card_acceptance supported_cards_and_currencies visa_or_mastercard""",
    "payments_transfers": """cancel_transfer declined_transfer failed_transfer pending_transfer transfer_into_account
        transfer_not_received_by_recipient transfer_timing beneficiary_not_allowed
        balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit
        receiving_money automatic_top_up pending_top_up top_up_failed top_up_limits top_up_reverted
        top_up_by_cash_or_cheque topping_up_by_card verify_top_up declined_card_payment
        pending_card_payment declined_cash_withdrawal pending_cash_withdrawal""",
    "billing_fees": """Refund_not_showing_up request_refund reverted_card_payment? transaction_charged_twice
        extra_charge_on_statement card_payment_fee_charged card_payment_not_recognised cash_withdrawal_charge
        cash_withdrawal_not_recognised direct_debit_payment_not_recognised exchange_charge exchange_rate
        exchange_via_app card_payment_wrong_exchange_rate wrong_exchange_rate_for_cash_withdrawal
        wrong_amount_of_cash_received transfer_fee_charged top_up_by_bank_transfer_charge
        top_up_by_card_charge fiat_currency_support country_support""",
    "account_access": """passcode_forgotten edit_personal_details terminate_account unable_to_verify_identity
        verify_my_identity verify_source_of_funds why_verify_identity age_limit lost_or_stolen_phone""",
}
TEAM_OF = {i: t for t, s in TEAMS.items() for i in s.split()}
assert len(TEAM_OF) == 77, len(TEAM_OF)  # every banking77 intent in exactly one team

CHOICES = {
    "card_issues": "physical or virtual cards: delivery, activation, PIN, lost or stolen, not working",
    "payments_transfers": "transfers, top-ups, deposits and declined or pending payments and withdrawals",
    "billing_fees": "unexpected charges, fees, refunds, exchange rates and disputed transactions",
    "account_access": "identity verification, personal details, passcode and closing the account",
    "other": "anything that is not a banking support request",
}
TASK = "Route the banking support message to exactly one team."


def build(bank, oos, n_per_team, n_other):
    parts = [bank[bank.team == t].sample(n_per_team, random_state=SEED) for t in TEAMS]
    rows = [(r.text, r.team) for p in parts for r in p.itertuples()]
    rows += [(t, "other") for t in oos.sample(n_other, random_state=SEED).text]
    return [{"task": TASK, "text": t, "choices": CHOICES, "label": l}
            for t, l in pd.Series(dict(enumerate(rows))).sample(frac=1, random_state=SEED)]


# Second domain: AG News topics. `other` = CLINC out-of-scope chit-chat, i.e. NOT a news item.
A = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/{}-00000-of-00001.parquet"
SECTIONS = ["world", "sports", "business", "sci_tech"]  # AG News label ids 0..3
NEWS_CHOICES = {
    "world": "international news, politics, government and conflicts",
    "sports": "sports results, teams, athletes and competitions",
    "business": "companies, markets, economy, finance and trade",
    "sci_tech": "science, technology, computing, the internet and space",
    "other": "not a news article (chit-chat, a question or a command to an assistant)",
}
NEWS_TASK = "Route the news article to exactly one section."


def build_news(news, oos, n_per_class, n_other):
    rows = [(r.text, SECTIONS[r.label]) for i in range(4) for r in news[news.label == i].sample(n_per_class, random_state=SEED).itertuples()]
    rows += [(t, "other") for t in oos.sample(n_other, random_state=SEED).text]
    return [{"task": NEWS_TASK, "text": t, "choices": NEWS_CHOICES, "label": l}
            for t, l in pd.Series(dict(enumerate(rows))).sample(frac=1, random_state=SEED)]


TRAIN_N = 150  # per class (incl. `other`); fine-tuning only, never used for thresholds or reported numbers


def dev_texts(path):
    return {json.loads(l)["text"] for l in open(path)}


def main():
    tr, te = (pd.read_json(B.format(s), lines=True) for s in ("train", "test"))
    for d in (tr, te):
        d["team"] = d.label_text.map(TEAM_OF)
    oos = {s: (lambda d: d[d.intent == OOS])(pd.read_parquet(C.format(s))) for s in ("validation", "test")}
    for name, bank, o, n, k in (("dev", tr, oos["validation"], 50, 50), ("test", te, oos["test"], 25, 25)):
        with open(f"data/support_{name}.jsonl", "w") as f:
            for r in build(bank, o, n, k):
                f.write(json.dumps(r) + "\n")
        print("support", name, 4 * n + k)
    ntr, nte = (pd.read_parquet(A.format(s)) for s in ("train", "test"))
    for name, news, o, n, k in (("dev", ntr, oos["validation"], 50, 50), ("test", nte, oos["test"], 25, 25)):
        with open(f"data/news_{name}.jsonl", "w") as f:
            for r in build_news(news, o, n, k):
                f.write(json.dumps(r) + "\n")
        print("news", name, 4 * n + k)

    # Train splits for the fine-tuned model: same sources' train parts, minus anything that is in dev.
    oos_train = pd.read_parquet(C.format("train"))
    oos_train = oos_train[oos_train.intent == OOS]
    for dom, src, builder in (("support", tr, lambda d, o: build(d, o, TRAIN_N, TRAIN_N)),
                              ("news", ntr, lambda d, o: build_news(d, o, TRAIN_N, TRAIN_N))):
        seen = dev_texts(f"data/{dom}_dev.jsonl")
        rows = builder(src[~src.text.isin(seen)], oos_train)
        assert not {r["text"] for r in rows} & seen
        with open(f"data/{dom}_train.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(dom, "train", len(rows))


if __name__ == "__main__":
    main()
