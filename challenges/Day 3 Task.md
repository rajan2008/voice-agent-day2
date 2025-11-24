# ✅ Day 3 – Order Review & Payment Agent

For Day 3, your main objective is to extend your Day 2 barista agent so it can review the final order, handle edits, and simulate payment confirmation before completing the workflow.

🎯 Primary Goal (Required)
✅ Persona

Keep the same friendly barista persona—or enhance it by making the agent more customer-service oriented (warm, polite, helpful).

✅ Order Review Behavior

Once the order state is complete, the agent should:

Clearly summarize the order back to the customer.

Ask: “Would you like to make any changes?”

If the customer wants changes:

Update the relevant fields in the order state.

Reconfirm the updated order.

If everything looks good:

Ask for a payment confirmation (e.g., “Shall I proceed to payment?”).

On confirmation, mark the order as complete.

✅ State Object (Same as Day 2)
{
  "drinkType": "string",
  "size": "string",
  "milk": "string",
  "extras": ["string"],
  "name": "string"
}


You will continue using this object, but now you’ll support editing the fields.

✅ Final Output

When payment is confirmed:

Save the final confirmed order into a JSON file named like:

order_<timestamp>.json


Log an internal message like:

Order completed and saved successfully.

🧩 Key Behaviors

✅ Must allow order changes
✅ Must reconfirm order after edits
✅ Must only save once the customer approves
✅ Must mark order as "completed"

🔧 Resources

https://docs.livekit.io/agents/build/tools/

https://docs.livekit.io/agents/build/agents-handoffs/#passing-state

https://docs.livekit.io/agents/build/tools/storage/

(You’ll mainly use state updates + file writing.)

✅ What Counts as Completed for Day 3

Your agent can:

Review the order

Handle changes

Confirm payment

Save the final JSON

You connect in the browser and successfully go through:
✅ Order
✅ Review
✅ Edit (optional)
✅ Final confirmation
✅ JSON saved

Record a short video showing:

The conversation

The saved JSON file

Post on LinkedIn with:
✅ What you built on Day 3
✅ Mention Murf Falcon (fastest TTS API)
✅ Mention “Murf AI Voice Agent Challenge”
✅ Tag official Murf AI handle
✅ Use hashtags:

#MurfAIVoiceAgentsChallenge
#10DaysofAIVoiceAgents