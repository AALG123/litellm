import type { ComplexityTiers } from "./ComplexityRouterConfig";

/**
 * A complexity tier maps to `str | list[str]` on the backend
 * (litellm/router_strategy/complexity_router/config.py: "string = pin; list = random pick"),
 * and the router widens the bare string with `models if isinstance(models, list) else [models]`.
 *
 * Every UI reader of a STORED complexity_router_config must widen the same way, so this is the
 * single owner of that rule. Readers of in-memory ComplexityTiers state are already string[]
 * and do not need it.
 */
export const normalizeTierModels = (value: unknown): string[] => {
  if (Array.isArray(value)) return value.filter((model): model is string => typeof model === "string");
  if (typeof value === "string" && value) return [value];
  return [];
};

/**
 * The deployment's default model, which the backend uses both when the resolved tier's pool is
 * empty (_get_model_for_tier) and as the classifier_fallback='default_model' destination.
 *
 * `pinned` is an explicit choice that beats the tiers; without one the value tracks the tiers in
 * the same MEDIUM-first order router.py falls back to. Every UI site that needs the default model -
 * submit, save, the routing test, and the fallback radio's enabled state - resolves it here, so the
 * value the caller is shown is the value that gets written to complexity_router_default_model.
 */
export const resolveComplexityDefaultModel = (tiers: ComplexityTiers, pinned?: string): string | undefined =>
  pinned?.trim() || tiers.MEDIUM[0] || tiers.SIMPLE[0] || tiers.COMPLEX[0] || tiers.REASONING[0];
