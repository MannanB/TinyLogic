import sentencepiece as spm


sp = spm.SentencePieceProcessor(model_file="tlm_bpe_8k.model")
print(sp.encode("What is the 59 + 284 / 194599 * 6996 sin(2) deg - +-=-=", out_type=str))

print(sp.encode("""The Kola Peninsula (Russian: Ко́льский полуо́стров, romanized: Kolsky poluostrov; Kildin Sami: Куэлнэгк нёа̄ррк) is a peninsula in the extreme northwest of Russia, and one of the largest peninsulas of Europe. Constituting the bulk of the territory of Murmansk Oblast, it lies almost completely inside the Arctic Circle and is bordered by the Barents Sea to the north and by the White Sea to the east and southeast. The city of Murmansk, the most populous settlement on the peninsula, has a population of roughly 270,000 residents.[1]

While humans had already settled in the north of the peninsula in the 7th–5th millennium BC, the rest of its territory remained uninhabited until the 3rd millennium BC, when various peoples started to arrive from the south. By the 1st millennium CE only the Sami people remained. This changed in the 12th century, when Russian Pomors discovered the peninsula's rich resources of game and fish. Soon after, the Pomors were followed by the tribute collectors from the Novgorod Republic, and the peninsula gradually became a part of the Novgorodian lands. However, the Novgorodians established no permanent settlements until the 15th century, and Russian migration continued in the following centuries. """, out_type=str))

print(sp.vocab_size())