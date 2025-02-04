type GreetProps = {
  name: string;
  messageCount?: number;
  isLoggedIn: boolean;
};
export const Greet = (props: GreetProps) => {
  const { messageCount = 0 } = props;
  const { name } = props;
  return (
    <div>
      <h2>Welcome {name}! You have 10k messages</h2>
    </div>
  );
};
