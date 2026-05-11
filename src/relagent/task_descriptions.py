"""
Task descriptions from RelBench website.
These are the official, detailed task descriptions that should be used in prompts.
"""

# Task descriptions from https://relbench.stanford.edu/datasets/
# Format: (dataset_name, task_name) -> description
TASK_DESCRIPTIONS = {
    # rel-amazon tasks
    ("rel-amazon", "user-churn"): "For each user, predict 1 if the customer does not review any product in the next 3 months, and 0 otherwise.",
    ("rel-amazon", "item-churn"): "For each product, predict 1 if the product does not receive any reviews in the next 3 months.",
    ("rel-amazon", "user-ltv"): "For each user, predict the $ value of the total number of products they buy and review in the next 3 months.",
    ("rel-amazon", "item-ltv"): "For each product, predict the $ value of the total number purchases and reviews it recieves in the next 3 months.",

    # rel-stack tasks
    ("rel-stack", "user-engagement"): "For each user predict if a user will make any votes, posts, or comments in the next 3 months.",
    ("rel-stack", "user-badge"): "For each user predict if a user will receive a new badge in the next 3 months.",
    ("rel-stack", "post-votes"): "For each user post predict how many votes it will receive in the next 3 months",

    # rel-trial tasks
    ("rel-trial", "study-outcome"): "Predict if the trials will achieve its primary outcome (defined as p-value < 0.05).",
    ("rel-trial", "study-adverse"): "Predict the number of affected patients with severe advsere events/death for the trial.",
    ("rel-trial", "site-success"): "Predict the success rate of a trial site in the next 1 year.",

    # rel-f1 tasks
    ("rel-f1", "driver-dnf"): "For each driver predict if they will DNF (did not finish) a race in the next 1 month.",
    ("rel-f1", "driver-top3"): "For each driver predict if they will qualify in the top-3 for a race in the next 1 month.",
    ("rel-f1", "driver-position"): "Predict the average finishing position of each driver all races in the next 2 months.",

    # rel-hm tasks
    ("rel-hm", "user-churn"): "For each user, predict whether a customer will have no transactions in the next week.",
    ("rel-hm", "item-sales"): "For each article, predict the total sales (sum of prices) in the next week.",

    # rel-event tasks
    ("rel-event", "user-repeat"): "For each user, predict if they will attend an event (yes/maybe) in the next 7 days if they attended one in the last 14 days.",
    ("rel-event", "user-ignore"): "For each user, predict if they will ignore more than 2 event invitations in the next 7 days.",
    ("rel-event", "user-attendance"): "For each user, predict how many events they will respond yes/maybe to in the next 7 days.",

    # rel-avito tasks
    ("rel-avito", "user-visits"): "For each user, predict whether they will visit more than one ad in the next 4 days.",
    ("rel-avito", "user-clicks"): "For each user, predict whether they will click on more than one ad in the next 4 days.",
    ("rel-avito", "ad-ctr"): "Given that an ad will be clicked in the next 4 days, predict its click-through-rate.",

    # rel-arxiv tasks (RelBenchV2)
    ("rel-arxiv", "paper-citation"): "For each paper, predict whether it will receive at least one citation in the next 6 months.",
    ("rel-arxiv", "author-publication"): "For each author, predict how many papers they will publish in the next 6 months.",

    # rel-ratebeer tasks (RelBenchV2)
    ("rel-ratebeer", "beer-churn"): "For each beer, predict whether it will receive a rating in the next 90 days.",
    ("rel-ratebeer", "user-churn"): "For each user, predict whether they will give a beer rating in the next 90 days.",
    ("rel-ratebeer", "brewer-dormant"): "For each brewer, predict whether they will release zero new beers in the next 365 days.",
    ("rel-ratebeer", "user-count"): "For each user, predict the number of beer ratings they will give in the next 90 days.",
}


# ---------------------------------------------------------------------------
# 4DBInfer task descriptions
# ---------------------------------------------------------------------------

FOURDBINFER_TASK_DESCRIPTIONS = {
    ("amazon-4db", "user-churn"): (
        "Predict whether an Amazon reviewer will stop writing reviews in the next time window (churn). "
        "Binary classification: 1 = churned (no future reviews), 0 = active."
    ),
    ("outbrain-4db", "ad-ctr"): (
        "Predict whether a user will click on a promoted content recommendation (click-through rate). "
        "Binary classification on the Outbrain content recommendation platform."
    ),
    ("retailrocket-4db", "item-cvr"): (
        "Predict whether a user will convert (purchase) after viewing an item on the RetailRocket "
        "e-commerce platform. Binary classification: 1 = purchase, 0 = no purchase."
    ),
    ("stackexchange-4db", "post-upvote"): (
        "Predict whether a StackExchange post will receive an upvote. "
        "Binary classification based on post content and user activity features."
    ),
    ("stackexchange-4db", "user-churn"): (
        "Predict whether a StackExchange user will churn (stop posting or participating) "
        "in the next time window. Binary classification: 1 = churned, 0 = active."
    ),
}


def get_4dbinfer_task_description(dataset_name: str, task_name: str) -> str:
    """Get task description for a 4DBInfer dataset/task pair."""
    key = (dataset_name, task_name)
    if key in FOURDBINFER_TASK_DESCRIPTIONS:
        return FOURDBINFER_TASK_DESCRIPTIONS[key]
    return f"4DBInfer node classification task on {dataset_name}/{task_name}."


def get_task_description(dataset_name: str, task_name: str) -> str:
    """
    Get the official task description from the website.

    Args:
        dataset_name: Name of the dataset (e.g., "rel-amazon")
        task_name: Name of the task (e.g., "user-churn")

    Returns:
        Task description string

    Raises:
        KeyError: If the task description is not found in the mapping
    """
    key = (dataset_name, task_name)
    if key not in TASK_DESCRIPTIONS:
        raise KeyError(
            f"Task description not found for {dataset_name}/{task_name}. "
            f"Available tasks for {dataset_name}: {[k[1] for k in TASK_DESCRIPTIONS if k[0] == dataset_name]}"
        )
    return TASK_DESCRIPTIONS[key]
