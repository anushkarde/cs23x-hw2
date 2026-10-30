from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from inkling_renderer import get_renderer_name
import inspect
import asyncio

def build_config() -> train.Config:
    model_name = "thinkingmachines/Inkling-Small"
    renderer_name = get_renderer_name(model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path="results/synthetic_personalized_data.jsonl",
    )

    config_kwargs = dict(
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_builder=dataset,
        log_path="results/personalized_email_sft",
        learning_rate=2e-4,
        lr_schedule="linear",
        num_epochs=1,
        wandb_project="personalized-emails",
        wandb_name="anushka-email-sft",
    )

    # Newer cookbook versions also require a recipe name.
    if "recipe_name" in inspect.signature(train.Config).parameters:
        config_kwargs["recipe_name"] = "personalized_email_sft"

    return train.Config(**config_kwargs)
    

def main():
    load_dotenv()
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    result = train.main(config)
    if inspect.isawaitable(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()