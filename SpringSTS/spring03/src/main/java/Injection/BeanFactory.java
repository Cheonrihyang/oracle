package Injection;

public class BeanFactory {
	
	SonySpeaker speaker = new SonySpeaker();
	
	public Object getBean(String beanName) {
		if(beanName.equals("samsung")) {
			return new SamsungTV(speaker);
		}else if(beanName.equals("lg")) {
			return new LGTV(speaker);
		}
		return null;
	}
}
