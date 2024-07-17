package com.spring.view2.board;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.Controller;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;
import com.spring.biz.board.impl.BoardServiceImpl;

public class UpdateBoardController implements Controller{
	BoardServiceImpl service;
	public void setBoardService(BoardServiceImpl service) {
		this.service = service;
	}
	@Override
	public ModelAndView handleRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
		System.out.println("updateBoard.to 성공");
		
		String seq = request.getParameter("seq");
		String title = request.getParameter("title");
		String content = request.getParameter("content");
		
		BoardVO boardVO = new BoardVO();
		boardVO.setTitle(title);
		boardVO.setContent(content);
		boardVO.setSeq(Integer.parseInt(seq));
		
		//BoardDAO boardDAO = new BoardDAO();
		service.updateBoard(boardVO);
		
		ModelAndView mav = new ModelAndView();
		mav.setViewName("redirect:getBoardList.to");
		return mav;
	}

}
