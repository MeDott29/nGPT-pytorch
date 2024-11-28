import training_audio_feedback as TrainingAudioFeedback

# Initialize the audio feedback system
audio_feedback = TrainingAudioFeedback()

# Training loop
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model_a.train()
    model_b.train()

    # Training step for both models
    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        # Forward and backward passes for model A
        loss_a = model_a(data, return_loss=True)
        (loss_a / GRAD_ACCUM_EVERY).backward()

        # Forward and backward passes for model B
        loss_b = model_b(data, return_loss=True)
        (loss_b / GRAD_ACCUM_EVERY).backward()

    # Compute gradient norms
    grad_norm_a = compute_gradient_norm(model_a)
    grad_norm_b = compute_gradient_norm(model_b)

    # Update audio feedback
    audio_feedback.update_training_state(
        loss_a=loss_a.item(),
        loss_b=loss_b.item(),
        grad_norm_a=grad_norm_a,
        grad_norm_b=grad_norm_b
    )

    print(f"training loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

    # Update both models
    optim_a.step()
    optim_b.step()
    optim_a.zero_grad()
    optim_b.zero_grad()

    # Normalize weights for both models
    model_a.norm_weights_()
    model_b.norm_weights_()

    # Validation step
    if i % VALIDATE_EVERY == 0:
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss_a = model_a(valid_data, return_loss=True)
            loss_b = model_b(valid_data, return_loss=True)
            
            # Update audio feedback with validation losses
            audio_feedback.update_training_state(
                loss_a=loss_a.item(),
                loss_b=loss_b.item()
            )
            
            print(f"validation loss - Model A: {loss_a.item():.3f}, Model B: {loss_b.item():.3f}")

    # Generation step
    if i % GENERATE_EVERY == 0:
        model_a.eval()
        model_b.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"Prime text: {prime} \n\n {'*' * 100}")
        prompt = inp[None, ...]

        # Generate from both models
        print("\nGeneration from Model A (with value residual):")
        sampled_a = base_decoding(model_a, prompt, GENERATE_LENGTH)
        generated_text_a = decode_tokens(sampled_a[0])
        print(generated_text_a)

        print("\nGeneration from Model B (with value residual):")
        sampled_b = base_decoding(model_b, prompt, GENERATE_LENGTH)
        generated_text_b = decode_tokens(sampled_b[0])
        print(generated_text_b)

        # Provide audio feedback on generation quality
        audio_feedback.feedback_generation_quality(
            output_a=generated_text_a,
            output_b=generated_text_b
        )