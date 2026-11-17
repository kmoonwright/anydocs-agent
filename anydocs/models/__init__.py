"""Model access, by role.

`config/models.yaml` maps a role to a provider and a model; `chat.py` turns that into a LangChain
chat model and `roles.py` calls it. The role is what appears on spans, so experiments compare
roles rather than model strings.
"""
