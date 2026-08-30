"""Prompt rendering without making LangChain a hard import dependency."""

from __future__ import annotations


def format_prompt(template: str, values: dict, input_variables=None) -> str:
    """Render a template using LangChain when available."""

    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        try:
            from langchain.prompts import PromptTemplate
        except ImportError:
            return template.format(**values)

    return PromptTemplate(
        input_variables=input_variables or list(values),
        template=template,
    ).format(**values)


def render_prompt(prompt_config, values: dict) -> str:
    """Render a legacy prompt config using LangChain when available."""

    return format_prompt(
        prompt_config.template,
        values,
        input_variables=prompt_config.input_variables,
    )
