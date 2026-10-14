"""Canonical acceptance inputs; no private speech. SPDX-License-Identifier: MIT."""
BASE = {"format": "wav", "streaming": False, "max_new_tokens": 1024, "seed": 42, "temperature": 0.7, "top_p": 0.7, "chunk_length": 200, "single_chunk": False, "references": []}
TEXTS = {
    "english-1": "Hello, this is Drama three running on a single Spark. The morning test is now complete.",
    "english-2": "A quiet garden surrounds the old house. Today we will walk beside the river and watch the birds.",
    "german-1": "Guten Morgen. Dieses Sprachmodell läuft auf einem einzelnen Spark. Die Sonne scheint und die Vögel singen im Garten.",
    "german-2": "Heute gehen wir gemeinsam zum Bahnhof. Danach trinken wir einen Kaffee und lesen ein interessantes Buch.",
    "expression": "[happy] What a wonderful morning! I am so excited to hear the good news.",
    "streaming": "The first part of this message is arriving as audio. We are testing native speech generation on a single Spark computer. The second part continues the story. Outside the window, the birds are singing and the sky is clear. A friendly traveler walks down the street and stops at the bakery to buy a fresh loaf of bread. We continue speaking so that more than one native text chunk is generated. This final sentence marks the end of the streaming test.",
    "longer": "A traveler arrived at a small village beside the river. The streets were quiet, and warm light shone from the bakery. She bought a loaf of bread and asked the baker for directions. He pointed toward the bridge and described a path through the forest. Along the way, she saw bright flowers, tall trees, and a family of ducks swimming in the water. At last she reached the old library, where a friendly librarian welcomed her. They talked about books, maps, and the history of the village. Before sunset, the traveler returned to the inn and wrote a letter about her peaceful day.",
    "overlap-a": "The blue bicycle stands beside the river. This is the first overlapping request.",
    "overlap-b": "The red balloon floats above the garden. This is the second overlapping request.",
    "reference": "The art of printing has changed the way we share stories. This new sentence uses a public domain reference recording.",
    "post-restart": "After restarting the service, this fresh audio proves that Drama three is still working on the Spark.",
    "final": "Good morning. Drama three is ready on the Spark, and this is the final live publication check.",
}


def request_for(name):
    result = {**BASE, "references": [], "text": TEXTS[name]}
    if name == "streaming":
        result["streaming"] = True
    return result
