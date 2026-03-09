import langextract as lx


def extract_intelligence(input_text: str):
    """
    Uses LangExtract and a local Ollama model to pull nodes and edges.
    """
    prompt = """
    Extract geopolitical entities and the relationships between them in order of appearance.
    Use exact text for extractions. Do not paraphrase.
    - Entities should be classified as 'Country', 'Technology', or 'Event'.
    - Relationships should connect two entities (e.g., 'sanctions', 'invests in', 'disrupts').
    Provide meaningful attributes to add context.
    """

    examples = [
        lx.data.ExampleData(
            text="The US government announced new export restrictions on advanced AI chips to China.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="Country",
                    extraction_text="US",
                    attributes={"role": "instigator"},
                ),
                lx.data.Extraction(
                    extraction_class="Technology",
                    extraction_text="AI chips",
                    attributes={"status": "restricted"},
                ),
                lx.data.Extraction(
                    extraction_class="Country",
                    extraction_text="China",
                    attributes={"role": "target"},
                ),
                lx.data.Extraction(
                    extraction_class="Relationship",
                    extraction_text="export restrictions",
                    attributes={"source": "US", "target": "China"},
                ),
            ],
        )
    ]

    # Using Llama 3.2 via local Ollama
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="llama3.2",
    )

    return result.extractions
