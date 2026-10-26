/* The API validates every selected model before it creates an inference job. */
export async function submitCompatibleModels(models, submit, onRejected, isCurrent = () => true) {
  let pending = [...models];
  const skipped = [];
  while (pending.length && isCurrent()) {
    try {
      const accepted = await submit(pending);
      return {accepted, models: pending, skipped, cancelled: !isCurrent()};
    } catch (error) {
      if (!isCurrent()) return {accepted: null, models: [], skipped, cancelled: true};
      const id = error.details?.model_id;
      if (error.code !== "MISSING_REQUIRED_CHANNEL" || !pending.includes(id) || !Array.isArray(error.details?.channels)) throw error;
      // A rejected batch has not started a job. Retry only the other explicitly
      // selected models, using the server's own alias matching as the authority.
      const issue = {modelId: id, channels: error.details.channels.filter(channel => typeof channel === "string")};
      skipped.push(issue);
      onRejected(issue);
      pending = pending.filter(model => model !== id);
    }
  }
  return {accepted: null, models: [], skipped, cancelled: !isCurrent()};
}
