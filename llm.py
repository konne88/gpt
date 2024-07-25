from dataclasses import dataclass
from typing import Callable, Iterator, Generic, List, Tuple, TypeVar
from utils import normalize, prefixes, weightedAverage

Token = int
Embedding = TypeVar('Embedding')
Value = TypeVar('Value')


@dataclass
class AttentionHead(Generic[Embedding, Value]):
    score: Callable[[Embedding, Embedding], float]
    value: Callable[[Embedding], Value]


@dataclass
class TransformerLayer(Generic[Embedding, Value]):
    heads: List[AttentionHead[Embedding, Value]]
    process: Callable[[Embedding, List[Value]], Embedding]


@dataclass
class Transformer(Generic[Embedding, Value]):
    embed: Callable[[int, Token], Embedding]
    layers: List[TransformerLayer[Embedding, Value]]
    unembed: Callable[[Embedding], Token]


def attendTo(embeddings: List[Embedding],
             score: Callable[[Embedding, Embedding], float],
             value: Callable[[Embedding], Value]) -> Value:
    current = embeddings[-1]
    scores = [score(current, other) for other in embeddings]
    scores = normalize(scores)
    values = [value(other) for other in embeddings]
    return weightedAverage(scores, values)


def enrichEmbedding(layer: TransformerLayer[Embedding, Value], embeddings: List[Embedding]) -> Embedding:
    current = embeddings[-1]
    focused = [attendTo(embeddings, head.score, head.value)
               for head in layer.heads]
    return layer.process(current, focused)


def nextToken(transformer: Transformer[Embedding, Value], tokens: List[Token]) -> Token:
    embeddings = [transformer.embed(index, token)
                  for (index, token) in enumerate(tokens)]
    for layer in transformer.layers:
        embeddings = [enrichEmbedding(layer, prefix)
                      for prefix in prefixes(embeddings)]
    return transformer.unembed(embeddings[-1])


def autocomplete(transformer: Transformer[Embedding, Value], max_seq_len: int, tokens: List[Token]) -> Iterator[Token]:
    yield from tokens
    n_prompt_tokens = len(tokens)
    for _ in range(max_seq_len - n_prompt_tokens):
        token = nextToken(transformer, tokens)
        tokens.append(token)
        yield token
