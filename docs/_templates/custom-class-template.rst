{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == "pydantic_model" %}

.. autopydantic_model:: {{ name }}

{% else %}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :special-members: __init__

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {% if item not in inherited_members %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% endif %}
