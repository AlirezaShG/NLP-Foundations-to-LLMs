# TravelBot: evaluation across 14 conversational scenarios

Manual evaluation of the LangGraph tool-calling agent in
[`07_tool_calling_travel_agent.ipynb`](../notebooks/07_tool_calling_travel_agent.ipynb).
Fourteen scenarios were run end to end, seven of them in Persian, covering all
seven tools. The orchestrating model was **GPT-4o-mini**.

Each scenario was judged on two things separately: whether the agent selected
and parameterised the right tool, and whether the answer it produced was
actually useful.

## Flights — `search_flights`

**1. Tehran → Dubai.** Origin and destination resolved correctly (IKA → DXB)
and a sensible default date applied. Three flights returned with full detail.
**Good.** Notably it filtered out DWC (Al Maktoum), the unrelated Dubai
airport — the entity resolution is doing real work, not just string matching.

**2. Tokyo → New York, "next month".** The relative date was resolved to a
concrete one (14 March) correctly. **Adequate.** But the top suggestion was a
35-hour itinerary: results are ranked by price with no consideration of
duration. Ranking on total travel time, or surfacing both, would be the better
default.

## Restaurants — `search_restaurants`

**3. London.** Instead of naming one restaurant, the agent returned
established curated lists (Time Out and similar). **Reasonable** — for a city
with thousands of options, pointing at a maintained list is more defensible
than picking one arbitrarily.

**4. Cairo, asked in Persian.** The Persian input was parsed and the tool
called successfully. **Good.** Here it did name a specific venue (Bab Al Qasr)
with a full description, which is the better experience — the inconsistency
with scenario 3 is not driven by anything in the prompt, so the behaviour is
model-dependent rather than controlled.

## Hotels — `search_hotels`

**5. Canberra, "from the start to the end of next week".** The compound date
range was resolved correctly. **Good.** Returning guest ratings alongside a
spread of prices gives the user something to actually decide on.

**6. Dubai.** Stay duration computed correctly from the dates. **Weakest of the
three hotel/flight results.** Every option returned was luxury-tier, and the
agent presented the list without comment. It should either have flagged that
the results skew expensive or offered to filter by budget.

## Weather — `get_weather`

**7. Moscow, next week.** The agent recognised that a multi-day forecast
requires several calls and invoked the tool three times for different dates.
**Strong** — this is the only scenario requiring the agent to loop a tool on
its own initiative, and it did so without prompting.

**8. Tehran, right now.** Single call for the current date. **Good** — fast,
answered in Persian, and complete (temperature and humidity).

## Currency — `get_currency_rate`

**9. Qatari riyal → UAE dirham.** **Good.** The agent volunteered a worked
example (the cost of buying 5 riyals), which makes a bare exchange rate
concretely useful.

**10. Chinese yuan → Canadian dollar, asked in Persian.** Both currencies
resolved from their Persian country names. **Good.** It hedged the reverse
conversion with "approximately", which is the right thing to do for a rate that
moves.

## Itineraries — `plan_trip`

**11. Mexico, 5 days.** A five-day plan was produced. **Appealing but not
feasible.** It spans Mexico City to Cancún — roughly 1,600 km — with no
allowance for the travel time. The tool has no notion of distance.

**12. Australia, interest = nature.** **Well personalised** — the plan
correctly centred on rainforest and diving. **Same structural flaw:** the
distances between the proposed stops are not accounted for.

This is the clearest weakness in the system. Entity extraction and date
handling are solid, but `plan_trip` has no geographic reasoning at all, so it
produces itineraries that read well and cannot be executed.

## FAQ retrieval — `search_faq`

**13. Travel documents, asked in Persian.** The right tool was called, but the
log shows it returned a chunk about "Language Barriers" — unrelated to the
question. **Retrieval failure.** The vector store did not surface the relevant
passage for the Persian query.

The final answer was nonetheless correct, because the LLM recognised the
retrieved chunk was irrelevant and answered from its own knowledge instead.
Worth being precise about what that means: the *user-visible* output was fine,
but the retrieval component failed, and the model silently covered for it. A
pipeline evaluated only on final answers would have scored this as a pass.

**14. Same question, asked in English.** **Consistent** — the answer matched
the Persian version, which indicates the knowledge base itself is coherent and
the problem in scenario 13 is cross-lingual retrieval, not the content.

A fifteenth scenario was also run, deliberately requiring several tools in one
request; the agent chained them correctly.

## Conclusions

**What works.** Natural-language understanding and entity extraction are the
strongest part of the system — dates (including relative and compound ranges),
cities, airports, and currencies are resolved reliably, in both Persian and
English. Multi-step tool orchestration works, including the agent deciding on
its own to call a tool repeatedly.

**What does not.**

1. **`plan_trip` has no geographic reasoning.** Itineraries ignore distance
   between stops, which makes some of them unrealisable. This is the most
   substantive limitation.
2. **Cross-lingual FAQ retrieval is unreliable.** The Persian query missed. A
   stronger multilingual embedding model is the first thing to try.
3. **City and country mapping is string-based.** Embedding-based resolution
   would be more robust than the current lookup. It did not visibly fail in
   these fourteen scenarios, but the fragility is there.
4. **Ranking is not always the useful one.** Flights sort by price even when
   duration is what matters; hotel results are returned unfiltered by budget.
