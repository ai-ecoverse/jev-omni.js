import coffee from "./images/coffee.png?url";
import shapes from "./images/shapes-0.png?url";
import receipt from "./images/receipt-0.png?url";
import traffic from "./images/traffic-0.png?url";
import screen from "./images/screen-2.png?url";

export interface Preset { state: string; question: string; options: string[]; image?: string; imageName?: string }

/** Short decisions in the style of DecisionBench: a situation, one question, its options. Written for this demo. */
export const textPresets: Record<string, Preset> = {
  "Support ticket routing": {
    state: "Customer message: \"My order #88213 arrived two weeks late, the shoes are the wrong size, and I see two charges for it on my card statement.\"",
    question: "Which team should handle this ticket first?",
    options: ["returns: exchanges, refunds, wrong or damaged items", "shipping: delivery status, delays, lost packages", "billing: charges, invoices, payment problems"],
  },
  "Refund policy": {
    state: "Policy: refunds within 30 days of delivery; items must be unused. Order #4411: trail boots, delivered 2026-08-05, returned 2026-08-20, worn outdoors twice.",
    question: "Does the policy allow a refund for this order?",
    options: ["Yes", "No"],
  },
  "Meeting status": {
    state: "The meeting starts at 10 AM. It is now 9 AM.",
    question: "Has the meeting started?",
    options: ["Yes", "No"],
  },
  "Incident severity": {
    state: "Monitoring alert: checkout API error rate at 38% for the last 12 minutes in eu-west; payments are failing for about a third of customers. Other regions are normal.",
    question: "How severe is this incident?",
    options: ["SEV4: cosmetic, no customer impact", "SEV3: minor, a workaround exists", "SEV2: major, degraded for many customers", "SEV1: critical, full outage"],
  },
};

/** Questions from kev.js's vision-v1 set (images: Apache-2.0 generated, and CC0 for the coffee photo by Rachel
 * Michetti, from scikit-image's sample data). */
export const imagePresets: Record<string, Preset> = {
  "Image: photo (coffee)": { image: coffee, imageName: "coffee.png", state: "A photo.", question: "What drink is shown?", options: ["coffee", "orange juice", "water", "wine"] },
  "Image: counting shapes": { image: shapes, imageName: "shapes-0.png", state: "A picture of simple shapes.", question: "How many shapes are in the picture?", options: ["1", "2", "3", "4", "5", "6", "7", "8"] },
  "Image: receipt": { image: receipt, imageName: "receipt-0.png", state: "A photo of a shopping receipt.", question: "Was this paid in cash?", options: ["Yes", "No"] },
  "Image: traffic light": { image: traffic, imageName: "traffic-0.png", state: "A photo of a traffic light.", question: "Should a car approaching this light stop?", options: ["Yes", "No"] },
  "Image: app screen": { image: screen, imageName: "screen-2.png", state: "A screenshot from an online shop.", question: "What kind of screen is this?", options: ["login", "payment result", "settings", "task list"] },
};
